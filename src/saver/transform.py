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

from src.vision_model import EmissionsTrackingCallback
from src.utils.secondary_utils import compute_model_params
from src.saver.extract_features import extract_features

class BaseSaver:
    def __init__(self, train_dataset:torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 discard: int = 0,
                 output_folder: str = "boh",
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
        os.makedirs(os.path.join(self.output_folder, model_name), exist_ok=True)
        dataset_name = self.train_dataset.__class__.__name__.lower()
        self.path_name = f"{dataset_name}_discard_{self.discard}_{self.batch_size}_{self.learning_rate}"
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

    def obtain_data_features(self):
        features = extract_features(train_dataset=self.train_dataset,
                                    test_dataset=self.test_dataset,
                                    dataset_name=self.train_dataset.__class__.__name__.lower())
        print(features)

    def check_saving(self, accuracy:float=None):
        if self.number_of_epochs <= 0:
            raise ValueError("You should include a valid number of epochs!")
        if accuracy is None or not isinstance(accuracy, float):
            raise ValueError("You must insert the accuracy as a floating number!!")

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
        
    def start_and_get_csv_logger(self):
        self.profiler = FlopsProfiler(self.model)
        self.profiler.output_dir = self.output_dir
        logger = CSVLogger(self.output_dir, 'version_0')
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

import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.number_of_epochs=10
        self.learning_rate = 0.001
        self.batch_size = 32
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # Calcolo accuracy semplice
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # Calcolo accuracy semplice
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Dataset e transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Dataset ridotto
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                 download=True, transform=transform)

    val_dataset = datasets.MNIST(root='./data', train=False, 
                               transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                transform=transform)

    # # Subset piccolissimo per test
    # train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    # val_dataset = torch.utils.data.Subset(val_dataset, range(200))
    # test_dataset = torch.utils.data.Subset(test_dataset, range(200))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model
    model = LightningModel()

    # Callbacks

    # Saver setup
    saver = CVSaver(train_dataset,
                    test_dataset,
                    model)
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
    trainer.fit(model, train_loader, val_loader)

    # Testing
    test_result = trainer.test(model, test_loader)
    
    # Save results
    saver.stop_and_save(accuracy=test_result[0]['test_acc'])


