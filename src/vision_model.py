import os
import json
from copy import deepcopy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
import time
from codecarbon import EmissionsTracker
import eco2ai



class Classifier(pl.LightningModule):
    def __init__(self, model_name: str = "resnet18", num_classes: int = 10, lr: float = 1e-3):
        super(Classifier, self).__init__()
        if model_name == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
        elif model_name == "resnet101":
            self.model = torchvision.models.resnet101(pretrained=True)
        elif model_name == "alexnet":
            self.model = torchvision.models.alexnet(pretrained=True)
        elif model_name == "vgg16":
            self.model = torchvision.models.vgg16(pretrained=True)
        elif model_name == "squeezenet":
            self.model = torchvision.models.squeezenet1_0(pretrained=True)
        elif model_name == "efficientnet":
            self.model = torchvision.models.efficientnet_b0(pretrained=True)
        elif model_name == "vit":
            self.model = torchvision.models.vit_b_16(pretrained=True)
        elif model_name == "mobilenet":
            self.model = torchvision.models.mobilenet_v2(pretrained=True)
        elif model_name == "swin_transformer":
            self.model = torchvision.models.swin_t(pretrained=True)
        else:
            raise ValueError(
                "Model not supported. Available models are: resnet18, resnet101, alexnet, vgg16, squeezenet, efficientnet, vit, mobilenet, swin_transformer"
            )

        if model_name.startswith("resnet"):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == "alexnet" or model_name == "vgg16":
            self.model.classifier[-1] = nn.Linear(
                self.model.classifier[-1].in_features, num_classes
            )
        elif model_name == "squeezenet":
            self.model.classifier[1] = nn.Conv2d(
                512, num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
        elif model_name.startswith("efficientnet"):
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, num_classes
            )
        elif model_name.startswith("vit"):
            self.model.heads.head = nn.Linear(
                self.model.heads.head.in_features, num_classes
            )
        elif model_name == "mobilenet":
            self.model.classifier[-1] = nn.Linear(
                self.model.classifier[-1].in_features, num_classes
            )
        elif model_name == "swin_transformer":
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class EarlyExitAccuracy(pl.callbacks.Callback):
    def __init__(self, target_accuracy):
        super().__init__()
        self.target_accuracy = target_accuracy

    def on_validation_epoch_end(self, trainer, pl_module):
        val_accuracy = trainer.callback_metrics.get("val_acc")
        if val_accuracy and val_accuracy >= self.target_accuracy:
            trainer.should_stop = True
            print(f"Reached {val_accuracy * 100:.2f}% accuracy")


class EmissionsTrackingCallback(pl.Callback):
    def __init__(self, exp_name):
        self.emissions_per_epoch = []
        self.times_per_epoch = []
        self.exp_name = exp_name

    def on_train_epoch_start(self, trainer, pl_module):
        self.tracker = EmissionsTracker(
            log_level="critical",
            tracking_mode="process",
            output_dir=self.exp_name,
            measure_power_secs=30,
            api_call_interval=4,
            experiment_id=self.exp_name,
        )
        self.tracker.start()
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.start_time
        self.times_per_epoch.append(epoch_time)
        latest_emission = self.tracker.stop()
        self.emissions_per_epoch.append(latest_emission)
        epoch = trainer.current_epoch


class SecondTrackerCallback(pl.Callback):
    def __init__(self, exp_name):
        self.emissions_per_epoch = []
        self.times_per_epoch = []
        self.exp_name = exp_name

    def on_train_epoch_start(self, trainer, pl_module):
        self.tracker = eco2ai.Tracker(
            project_name="Env footprint",
            file_name=f"{self.exp_name}/emission_eco2ai.csv",
            cpu_processes="current",
            measure_period=30,
            ignore_warnings=True,
            alpha_2_code="IT",
        )
        self.tracker.start()
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.start_time
        self.times_per_epoch.append(epoch_time)
        latest_emission = self.tracker.stop()
        self.emissions_per_epoch.append(latest_emission)
        epoch = trainer.current_epoch


