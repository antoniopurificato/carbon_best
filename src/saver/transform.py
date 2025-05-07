import torch
import os
from deepspeed.profiling.flops_profiler import FlopsProfiler
from pytorch_lightning.loggers import CSVLogger
import torch
from codecarbon import EmissionsTracker
import time
import yaml
import cpuinfo
import json
import pytorch_lightning as pl

from datasets import Dataset as HFDataset
import transformers

from src.vision_model import EmissionsTrackingCallback
from src.utils.secondary_utils import compute_model_params
from src.saver.extract_features import extract_features
from src.saver.save_to_google import upload_folder


class BaseSaver:
    def __init__(self, train_dataset:torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 discard: int = 0,
                 output_folder: str = "saved_experiments",
                 **kwargs
                 ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_folder = output_folder
        self.discard = discard

        self.check_properties()
        self.create_folders()
        self.extract_hardware_info()
        self.obtain_data_features()

    def check_lightning_module(self, lightning_module):
        if not hasattr(lightning_module, 'learning_rate'):
            raise ValueError("Your lightning module should have a learning_rate attribute!!")
        elif not hasattr(lightning_module, 'batch_size'):
            raise ValueError("Your lightning module should have a batch_size attribute!!")
        elif not hasattr(lightning_module, 'model'):
            raise ValueError("Your lightning module should have a model attribute!!")
        elif not hasattr(lightning_module, 'number_of_epochs'):
            raise ValueError("Your lightning module should have a number_of_epochs attribute!!")

    def create_folders(self):
        model_name = self.model.__class__.__name__.lower()
        initial_dir = os.path.join(self.output_folder, model_name)
        os.makedirs(initial_dir, exist_ok=True)
        self.dataset_name = self.train_dataset.__class__.__name__.lower()
        self.path_name = f"{self.dataset_name}_discard_{self.discard}_{self.batch_size}_{self.learning_rate}"
        self.output_dir = os.path.join(self.output_folder, model_name, self.path_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_hardware_info(self):
        properties = torch.cuda.get_device_properties(0)
        gpu_info = {key: getattr(properties, key) for key in dir(properties) if not key.startswith('_')}
        name = gpu_info['gcnArchName']
        self.hardware_mapping = {}
        self.hardware_mapping[name] = {'gpu' : gpu_info}
        del self.hardware_mapping[name]['gpu']['gcnArchName']
        self.hardware_mapping[name]['cpu'] = cpuinfo.get_cpu_info()
        with open(os.path.join(self.output_dir, 'hardware_info.json'), 'w', encoding='utf-8') as f:
            json.dump(self.hardware_mapping, f, ensure_ascii=False, indent=4)

    def check_saving(self, accuracy:float=None):
        if self.number_of_epochs <= 0:
            raise ValueError("You should include a valid number of epochs!")
        if accuracy is None or not isinstance(accuracy, float):
            raise ValueError("You must insert the accuracy as a floating number!!")

    def save_features(self, features:dict):
        with open(os.path.join(self.output_dir, "features.json"), "w") as json_file:
            json.dump(features, json_file)

    def upload_on_drive(self):
        upload_folder(folder=self.output_dir)

    def check_properties(self):
        raise NotImplementedError("Subclasses must implement check_properties")

    def start_and_get_csv_logger(self):
        raise NotImplementedError("Subclasses must implement start_and_get_csv_logger")

    def stop_and_save(self, accuracy:float=None, number_of_epochs:int=0):
        raise NotImplementedError("Subclasses must implement stop_and_save")

class CVSaver(BaseSaver):
    def __init__(self,
                 train_dataset:torch.utils.data.Dataset,
                 test_dataset:torch.utils.data.Dataset,
                 lightning_module:pl.LightningModule,
                 **kwargs):

        self.check_lightning_module(lightning_module)
        self.learning_rate = lightning_module.learning_rate
        self.batch_size = lightning_module.batch_size
        self.number_of_epochs = lightning_module.number_of_epochs
        self.model = lightning_module.model

        super().__init__(train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        **kwargs)

    def check_properties(self):
        if not isinstance(self.train_dataset, torch.utils.data.Dataset):
            raise ValueError("The dataset should be a torch.utils.data.Dataset!!")
        if not isinstance(self.test_dataset, torch.utils.data.Dataset):
            raise ValueError("The dataset should be a torch.utils.data.Dataset!!")
        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("The model should be a torch.nn.Module!!")
        if self.discard < 1 and self.discard != 0:
            raise ValueError("Discard must be a positive integer or zero.")
        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate should be a float!!")
        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size should be an integer!!")

    def obtain_data_features(self):
        features = extract_features(train_dataset=self.train_dataset,
                                    test_dataset=self.test_dataset,
                                    dataset_name=self.dataset_name)
        self.save_features(features)

    def start_and_get_csv_logger(self):
        self.profiler = FlopsProfiler(self.model)
        self.profiler.output_dir = self.output_dir
        logger = CSVLogger(self.output_dir)#, 'version_0')
        self.emissions_tracker = EmissionsTracker(
            tracking_mode="process",
            log_level="critical",
            output_dir=self.output_dir,
            measure_power_secs=30,
            api_call_interval=4,
            experiment_id=self.output_dir,
        )
        self.emissions_tracker.start()
        self.profiler.start_profile()
        self.experiment_start_time = time.time()
        self.emissions_callback = EmissionsTrackingCallback(self.output_dir)
        return logger, self.emissions_callback

    def stop_and_save(self, accuracy:float=None):

        self.check_saving(accuracy=accuracy)
        experiment_end_time = time.time()
        experiment_time = experiment_end_time - self.experiment_start_time
        self.total_emissions = self.emissions_tracker.stop()
        self.profiler.print_model_profile(
            output_file=f"{self.profiler.output_dir}/train_flops.txt"
        )
        self.profiler.stop_profile()
        with open("features.json", "r") as read_file:
            data = json.load(read_file)
        emissions_res = {}
        emissions_res[self.output_dir] = {
            "accuracy": accuracy,
            "num_params": compute_model_params(self.model),
            "flops": self.profiler.get_total_flops(),
            "epochs_concluded": self.number_of_epochs,
            "experiment_time": experiment_time,
            "total_emissions": self.total_emissions,
            "emissions_per_epoch": self.emissions_callback.emissions_per_epoch,
            "times_per_epoch": self.emissions_callback.times_per_epoch,
            "original_data_size": len(self.train_dataset),
            "modified_data_size": len(self.train_dataset) / self.discard if self.discard != 0.0  else len(self.train_dataset)
        }

        with open(os.path.join(self.output_dir, "results.yml"), "w") as yaml_file:
            yaml.dump(emissions_res[self.output_dir], yaml_file, default_flow_style=False)
        self.upload_on_drive()

class NLPSaver(BaseSaver):
    def __init__(self,
                 train_dataset: HFDataset,
                 test_dataset: HFDataset,
                 lightning_module: pl.LightningModule,
                 label_column:str=None,
                 **kwargs):

        self.check_lightning_module(lightning_module)
        self.learning_rate = lightning_module.learning_rate
        self.batch_size = lightning_module.batch_size
        self.number_of_epochs = lightning_module.number_of_epochs
        self.model = lightning_module.model
        self.label_column = label_column

        super().__init__(train_dataset=train_dataset,
                         test_dataset=test_dataset,
                         **kwargs)

    def check_properties(self):
        if not isinstance(self.train_dataset, HFDataset):
            raise ValueError("The dataset should be a huggingface datasets.Dataset!!")
        if not isinstance(self.test_dataset, HFDataset):
            raise ValueError("The dataset should be a huggingface datasets.Dataset!!")
        if self.discard < 1 and self.discard != 0:
            raise ValueError("Discard must be a positive integer or zero.")
        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate should be a float!!")
        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size should be an integer!!")

    def obtain_data_features(self):
        features = extract_features(train_dataset=self.train_dataset,
                                    test_dataset=self.test_dataset,
                                    dataset_name=self.dataset_name,
                                    label_column=self.label_column)
        self.save_features(features)

    def start_and_get_csv_logger(self):
        self.profiler = FlopsProfiler(self.model)
        self.profiler.output_dir = self.output_dir
        logger = CSVLogger(self.output_dir)#, 'version_0')
        self.emissions_tracker = EmissionsTracker(
            tracking_mode="process",
            log_level="critical",
            output_dir=self.output_dir,
            measure_power_secs=30,
            api_call_interval=4,
            experiment_id=self.output_dir,
        )
        self.emissions_tracker.start()
        self.profiler.start_profile()
        self.experiment_start_time = time.time()
        self.emissions_callback = EmissionsTrackingCallback(self.output_dir)
        return logger, self.emissions_callback

    def stop_and_save(self, accuracy: float = None):

        self.check_saving(accuracy=accuracy)
        experiment_end_time = time.time()
        experiment_time = experiment_end_time - self.experiment_start_time
        self.total_emissions = self.emissions_tracker.stop()
        self.profiler.print_model_profile(
            output_file=f"{self.profiler.output_dir}/train_flops.txt"
        )
        self.profiler.stop_profile()

        emissions_res = {}
        emissions_res[self.output_dir] = {
            "accuracy": accuracy,
            "num_params": compute_model_params(self.model),
            "flops": self.profiler.get_total_flops(),
            "epochs_concluded": self.number_of_epochs,
            "experiment_time": experiment_time,
            "total_emissions": self.total_emissions,
            "emissions_per_epoch": self.emissions_callback.emissions_per_epoch,
            "times_per_epoch": self.emissions_callback.times_per_epoch,
            "original_data_size": len(self.train_dataset),
            "modified_data_size": len(self.train_dataset) / self.discard if self.discard != 0.0 else len(
                self.train_dataset)
        }

        with open(os.path.join(self.output_dir, "results.yml"), "w") as yaml_file:
            yaml.dump(emissions_res[self.output_dir], yaml_file, default_flow_style=False)
        self.upload_on_drive()


# ------Example ------
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class LightningNLPModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.number_of_epochs = 3
        self.learning_rate = 2e-5
        self.batch_size = 16

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch['input_ids'])
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch['input_ids'])
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == batch['label']).float().mean()
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(batch['input_ids'])
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == batch['label']).float().mean()
        self.log("test_acc", correct / len(predictions))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    # Carica il dataset
    dataset = load_dataset("imdb")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


    def preprocess_function(examples):
        return tokenizer(examples['text'],
                         truncation=True,
                         padding='max_length',
                         max_length=512)


    # Preprocessa i dataset
    tokenized_train = dataset['train'].map(preprocess_function, batched=True)
    tokenized_test = dataset['test'].map(preprocess_function, batched=True)

    # Converti in formato PyTorch
    tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Subset piccolissimo per test
    tokenized_train = tokenized_train.select(range(10))
    tokenized_test = tokenized_test.select(range(2))

    # DataLoaders
    train_loader = DataLoader(tokenized_train, batch_size=16, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=16)

    # Model
    model = LightningNLPModel()

    # Saver setup
    saver = NLPSaver(tokenized_train,
                     tokenized_test,
                     model,
                     label_column='text')
    logger, emissions_callback = saver.start_and_get_csv_logger()

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='auto',
        devices=1,
        logger=logger,
        callbacks=[emissions_callback],
        enable_progress_bar=True,
        log_every_n_steps=1
    )

    # Training
    trainer.fit(model, train_loader)

    # Testing
    test_result = trainer.test(model, test_loader)

    # Save results
    saver.stop_and_save(accuracy=test_result[0]['test_acc'])