import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleCell(nn.Module):
    def __init__(self, adjacency_matrix, operations, input_channels, output_channels):
        super(SingleCell, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.operations = operations

        # Create layers for each operation in the cell
        self.node_layers = nn.ModuleList()
        for op in operations:
            if op == "input":
                self.node_layers.append(None)  # Input doesn't need a layer
            elif op == "output":
                self.node_layers.append(None)  # Output is just an aggregation node
            elif op == "conv3x3-bn-relu":
                self.node_layers.append(nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU()
                ))
            elif op == "conv1x1-bn-relu":
                self.node_layers.append(nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU()
                ))
            elif op == "maxpool3x3":
                self.node_layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            else:
                raise ValueError(f"Unsupported operation: {op}")

    def forward(self, x):
        node_outputs = [None] * len(self.operations)
        node_outputs[0] = x  # Input node output is the input tensor

        for i in range(1, len(self.operations)):
            input_sum = 0
            for j in range(i):  # Aggregate outputs from connected nodes
                if self.adjacency_matrix[j][i] == 1:
                    input_sum += node_outputs[j]

            # Pass the sum of inputs through the node's operation
            if self.operations[i] == "output":
                node_outputs[i] = input_sum  # Output node collects all outputs
            else:
                node_outputs[i] = self.node_layers[i](input_sum)

        return node_outputs[-1]  # Return the output of the final node


class NASBench101FullModel(nn.Module):
    def __init__(self, adjacency_matrix, operations, input_channels=224, num_classes=10):
        super(NASBench101FullModel, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Initial stem layer to process input
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU()
        )

        self.cells = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()

        # Build 3 blocks, each consisting of 3 cells followed by a downsampling layer
        for block_idx in range(3):
            for cell_idx in range(3):  # Stack 3 cells per block
                self.cells.append(SingleCell(
                    adjacency_matrix, 
                    operations, 
                    input_channels, 
                    input_channels
                ))
            # Downsampling layer after every 3 cells
            self.downsampling_layers.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # Halve height and width
                nn.Conv2d(input_channels, input_channels * 2, kernel_size=1),
                nn.BatchNorm2d(input_channels * 2),
                nn.ReLU()
            ))
            input_channels *= 2  # Double the number of channels after downsampling

        # Global average pooling and dense softmax layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)

        # Pass through cells and downsampling layers
        cell_idx = 0
        for block_idx in range(3):
            for _ in range(3):  # Each block has 3 cells
                x = self.cells[cell_idx](x)
                cell_idx += 1
            x = self.downsampling_layers[block_idx](x)  # Downsample after every block

        # Global average pooling and dense softmax layer
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        
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


class LitNASBench101(pl.LightningModule):
    def __init__(self, adjacency_matrix, operations, input_channels=224, num_classes=10, lr=10e-3):
        super().__init__()
        self.save_hyperparameters()  # Optional: saves hyperparameters
        self.model = NASBench101FullModel(
            adjacency_matrix,
            operations,
            input_channels,
            num_classes
        )
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