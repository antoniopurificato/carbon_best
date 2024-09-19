import torch
import os
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import argparse
import deepspeed
from torch.utils.data import DataLoader
import pandas as pd
from deepspeed.profiling.flops_profiler import FlopsProfiler
import yaml
from functools import partial
import optuna



from src.vision_model import *
from src.utils import *
from src.data import *

#TODO: automate all the exps


# deepspeed.init_distributed(dist_backend=None, distributed_port=29499)

def objective(
    trial,
    d_name: str = "cifar10",
    m_name: str = "resnet18",
    samples_to_discard: float = 0.0,
    seed: int = 42,
    gpu_id: int = 2
) -> dict:
    emissions_res = {}
    n_epochs = int(trial.suggest_float('n_epochs', 10, 100))
    lr = trial.suggest_float('lr', 1e-6, 1e-1)


    dataset, input_channels, num_classes = get_dataset(d_name)
    original_data_size = len(dataset)
    dataset = remove_samples(dataset, samples_to_discard)
    train_size = int(0.7 * len(dataset))
    dataset, val_dataset = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    val_size = int(0.66 * len(val_dataset))
    val_dataset, test_dataset = random_split(
        val_dataset,
        [val_size, len(val_dataset) - val_size],
        generator=torch.Generator().manual_seed(seed),
    ) 
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10)
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=10
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=10
    )

    model = Classifier(model_name=m_name, num_classes=num_classes, lr=lr)
    profiler = FlopsProfiler(model)

    # for n_epochs in epochs:
    perc_value = int(samples_to_discard*100)
    exp_name = f"{d_name}_discard_{perc_value}_{str(n_epochs)}_{str(lr)}"
    folder_path = "results/" + m_name
    exp_path = os.path.join(folder_path, exp_name)

    if not os.path.isfile(os.path.join(exp_path,'results.yml')):
        print(f"Experiment: {exp_path}")
        if not os.path.isdir(exp_path):
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            os.mkdir(exp_path)

        profiler.output_dir = exp_path

        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
        early_stopping_callback = EarlyExitAccuracy(target_accuracy=0.88)
        # flops_callback = FLOPsCallback(input_size=(1, input_channels, 224, 224))
        emissions_callback = EmissionsTrackingCallback(exp_path)
        emissions_eco2ai_callback = SecondTrackerCallback(exp_path)
        logger = CSVLogger(save_dir=folder_path, name=exp_name)

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            accelerator="gpu",
            devices=[gpu_id],
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                emissions_callback,
                emissions_eco2ai_callback,
            ],  # flopscallback, emissioncallback
            logger=logger,
        )  # per colab accelerator="gpu" if torch.cuda.is_available() else "cpu"
        total_tracker = EmissionsTracker(
            tracking_mode="process",
            log_level="critical",
            output_dir=exp_path,
            measure_power_secs=30,
            api_call_interval=4,
            experiment_id=exp_path,
        )
        eco2ai_tracker = eco2ai.Tracker(
            project_name="Env footprint",
            file_name=f"{exp_path}/emission_eco2ai.csv",
            cpu_processes="current",
            measure_period=30,
            ignore_warnings=True,
            alpha_2_code="IT",
        )
        eco2ai_tracker.start()
        total_tracker.start()
        profiler.start_profile()

        experiment_start_time = time.time()
        trainer.fit(model, train_loader, val_loader)
        experiment_end_time = time.time()
        experiment_time = experiment_end_time - experiment_start_time

        accuracy = trainer.test(model, test_loader)

        total_emissions = total_tracker.stop()
        profiler.print_model_profile(
            output_file=f"{profiler.output_dir}/train_flops.txt"
        )
        eco2ai_tracker.stop()
        emissions_eco2ai = float(pd.read_csv(f"{exp_path}/emission_eco2ai.csv").iloc[0]['CO2_emissions(kg)'])
        profiler.stop_profile()

        emissions_res[exp_path] = {
            "accuracy": accuracy[0]["test_acc"],
            "num_params": compute_model_params(
                trainer.model
            ),
            "flops": profiler.get_total_flops(), #check
            "epochs_concluded": trainer.current_epoch,
            "experiment_time": experiment_time,
            "total_emissions": total_emissions,
            "emissions_per_epoch": emissions_callback.emissions_per_epoch,
            "times_per_epoch": emissions_callback.times_per_epoch,
            "emissions_eco2ai": emissions_eco2ai,
            "original_data_size" : original_data_size,
            "modified_data_size" : len(dataset),
        }

        with open(os.path.join(exp_path, "results.yml"), "w") as yaml_file:
            yaml.dump(
                emissions_res[exp_path], yaml_file, default_flow_style=False
            )
        return accuracy[0]["test_acc"]  
    else:
        print(f"Experiment {exp_path} already exists") 
        return ValueError("Experiment already exists")


                    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=get_all_datasets(),
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[5,10,20,50,100],
        help="Number of epochs to train the model",
    )
    parser.add_argument("--gpu_id", type=int, default=2, help="Id of the GPU")
    parser.add_argument("--seed", type=int, default=42, help="Seed value")
    parser.add_argument(
        "--model", type=str, nargs="+", default=get_models(), help="Name of the model"
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    
    dataset_name=args.dataset
    model_name=args.model

    for d_name in dataset_name:
        for samples_to_discard in [1, .3, .5, .7, .1]:  
            for m_name in model_name:
                d = d_name
                m = m_name
                s = samples_to_discard
                study = optuna.create_study()
                objective = partial(objective, d_name = d, m_name = m, samples_to_discard = s, seed = args.seed, gpu_id = args.gpu_id)
                study.optimize(objective, n_trials = 15)
   