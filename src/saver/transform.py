import torch
import os
from deepspeed.profiling.flops_profiler import FlopsProfiler
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from codecarbon import EmissionsTracker
import time
import yaml
import cpuinfo
import json

from src.vision_model import EmissionsTrackingCallback
from src.utils.secondary_utils import compute_model_params

class Saver:
    def __init__(self, dataset:torch.utils.data.Dataset, 
                 model:torch.nn.Module,
                 learning_rate:float,
                 batch_size:int,
                 discard:int=0,
                 output_folder:str="boh"):
        self.dataset = dataset
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_folder = output_folder
        self.discard = discard

        self.check_properties()
        
        self.create_folders()
        self.extract_hardware_info()

    def check_properties(self):
        if not isinstance(self.dataset, torch.utils.data.Dataset):
            raise ValueError("The dataset should be a torch.utils.data.Dataset!!")
        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("The model should be a torch.nn.Module!!")
        if self.discard < 1 and self.discard != 0:
            raise ValueError("Discard must be a positive integer or zero.")
        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate should be a float!!")
        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size should be an integer!!")
    
    def check_saving(self, accuracy:float=None, number_of_epochs:int=0):
        if number_of_epochs == 0:
            raise ValueError("You should include the number of epochs!")
        if accuracy is None or not isinstance(accuracy, float):
            raise ValueError("You must insert the accuracy as a floating number!!")
    
    def create_folders(self):
        model_name = self.model.__class__.__name__.lower()
        os.makedirs(os.path.join(self.output_folder, model_name), exist_ok=True)
        dataset_name = self.dataset.__class__.__name__.lower()
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

    def stop_and_save(self, accuracy:float=None, number_of_epochs:int=0):
        
        self.check_saving(accuracy=accuracy, number_of_epochs=number_of_epochs)
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
            "epochs_concluded": number_of_epochs,
            "experiment_time": experiment_time,
            "total_emissions": self.total_emissions,
            "emissions_per_epoch": self.emissions_callback.emissions_per_epoch,
            "times_per_epoch": self.emissions_callback.times_per_epoch,
            "original_data_size": len(self.dataset),
            "modified_data_size": len(self.dataset) / self.discard if self.discard != 0.0  else len(self.dataset)
        }

        with open(os.path.join(self.output_dir, "results.yml"), "w") as yaml_file:
            yaml.dump(emissions_res[self.output_dir], yaml_file, default_flow_style=False)
        

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.network(x)
    
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
        return torch.optim.Adam(self.parameters(), lr=0.001)

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

    # Subset piccolissimo per test
    train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    val_dataset = torch.utils.data.Subset(val_dataset, range(200))
    test_dataset = torch.utils.data.Subset(test_dataset, range(200))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model
    model = LightningModel()

    # Callbacks

    # Saver setup
    saver = Saver(train_dataset, model, learning_rate=0.001, batch_size=32)
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
    saver.stop_and_save(test_result[0]['test_acc'])