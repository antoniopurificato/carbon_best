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

from src.vision_model import *
from src.utils.secondary_utils import *
from src.data import *

# It could be needed for problems with DeepSpeed initialization
# deepspeed.init_distributed(dist_backend=None, distributed_port=29499)


def start_exp(
    d_name: str = "cifar10",
    m_name: str = "resnet18",
    lr: float = 1e-4,
    n_epochs: int = 100,
    samples_to_discard: float = 0.0,
    seed: int = 42,
    gpu_id: int = 2,
    batch_size: int = 64,
) -> dict:
    """
    Run a training experiment for a given dataset and model configuration.

    Args:
        d_name (str): Name of the dataset to use (e.g., "cifar10").
        m_name (str): Name of the model to train (e.g., "resnet18").
        lr (float): Learning rate for training.
        n_epochs (int): Number of epochs for training.
        samples_to_discard (float): Proportion of dataset samples to discard (0.0 to 1.0).
        seed (int): Random seed for reproducibility.
        gpu_id (int): ID of the GPU to use.
        batch_size (int): Batch size for training and evaluation.

    Returns:
        dict: Dictionary containing experiment results, including accuracy, FLOPs, emissions, and runtime.

    """
    emissions_res = {}

    # Load dataset and preprocess
    dataset, input_channels, num_classes = get_dataset(d_name)
    original_data_size = len(dataset)
    dataset = remove_samples(dataset, samples_to_discard)  # Reduce dataset size

    # Split dataset into train, validation, and test sets
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

    # Initialize data loaders
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=10
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )

    # Initialize model
    model = Classifier(model_name=m_name, num_classes=num_classes, lr=lr)

    # Initialize FLOPs profiler
    profiler = FlopsProfiler(model)

    # Experiment naming and result folder setup
    perc_value = int(samples_to_discard * 100)
    exp_name = f"{d_name}_discard_{perc_value}_{str(batch_size)}_{str(lr)}"
    folder_path = "results/" + m_name
    exp_path = os.path.join(folder_path, exp_name)

    # Skip if the experiment has already been executed
    if not os.path.isfile(os.path.join(exp_path, "results.yml")):
        print(f"Experiment: {exp_path}")
        if not os.path.isdir(exp_path):
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            os.mkdir(exp_path)

        # Configure profiler output
        profiler.output_dir = exp_path

        # Define callbacks and logger
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
        emissions_callback = EmissionsTrackingCallback(exp_path)
        logger = CSVLogger(save_dir=folder_path, name=exp_name)

        # Configure PyTorch Lightning trainer
        trainer = pl.Trainer(
            max_epochs=n_epochs,
            accelerator="gpu",
            devices=[gpu_id],
            callbacks=[checkpoint_callback, emissions_callback],
            logger=logger,
        )

        # Start emissions tracker
        total_tracker = EmissionsTracker(
            tracking_mode="process",
            log_level="critical",
            output_dir=exp_path,
            measure_power_secs=30,
            api_call_interval=4,
            experiment_id=exp_path,
        )
        total_tracker.start()
        profiler.start_profile()

        # Measure experiment runtime
        experiment_start_time = time.time()
        trainer.fit(model, train_loader, val_loader)
        experiment_end_time = time.time()
        experiment_time = experiment_end_time - experiment_start_time

        # Evaluate model on test set
        accuracy = trainer.test(model, test_loader)

        # Stop emissions tracking and profiler
        total_emissions = total_tracker.stop()
        profiler.print_model_profile(
            output_file=f"{profiler.output_dir}/train_flops.txt"
        )
        profiler.stop_profile()

        # Save experiment results
        emissions_res[exp_path] = {
            "accuracy": accuracy[0]["test_acc"],
            "num_params": compute_model_params(trainer.model),
            "flops": profiler.get_total_flops(),
            "epochs_concluded": trainer.current_epoch,
            "experiment_time": experiment_time,
            "total_emissions": total_emissions,
            "emissions_per_epoch": emissions_callback.emissions_per_epoch,
            "times_per_epoch": emissions_callback.times_per_epoch,
            "original_data_size": original_data_size,
            "modified_data_size": len(dataset),
        }

        with open(os.path.join(exp_path, "results.yml"), "w") as yaml_file:
            yaml.dump(emissions_res[exp_path], yaml_file, default_flow_style=False)

        return accuracy[0]["test_acc"]
    else:
        print(f"Experiment {exp_path} already exists")


if __name__ == "__main__":
    """
    Main entry point for running multiple experiments with different configurations.
    """
    parser = argparse.ArgumentParser()

    # Command-line arguments for the script
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=get_all_datasets("vision"),
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[100],
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--bs",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="Values of batch size",
    )
    parser.add_argument(
        "--discard_percentage",
        type=float,
        nargs="+",
        default=[1, 0.3, 0.7],
        help="Values of discard percentage",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Id of the GPU")
    parser.add_argument("--seed", type=int, default=42, help="Seed value")
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=get_models("vision"),
        help="Name of the model",
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[10e-3, 10e-4, 10e-5],
        help="Learning rate values",
    )
    args = parser.parse_args()

    # Set the random seed for reproducibility
    seed_everything(args.seed)

    dataset_name = args.dataset
    model_name = args.model
    epochs = args.epochs
    counter = 0

    # Iterate through all combinations of parameters and run experiments
    for d_name in dataset_name:
        counter += 1
        for lr in args.lr:
            for samples_to_discard in args.discard_percentage:
                for batch_size in args.bs:
                    for m_name in model_name:
                        for epoch in epochs:
                            d = d_name
                            m = m_name
                            s = samples_to_discard
                            result = start_exp(
                                d_name=d,
                                m_name=m,
                                lr=lr,
                                n_epochs=epoch,
                                samples_to_discard=s,
                                seed=args.seed,
                                gpu_id=args.gpu_id,
                                batch_size=batch_size,
                            )
