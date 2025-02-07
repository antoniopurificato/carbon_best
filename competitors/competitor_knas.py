import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
import time
from codecarbon import EmissionsTracker

np.random.seed(0)

class NASNet(nn.Module):
    def __init__(self, arch_str, C=16, num_classes=10):
        super(NASNet, self).__init__()
        
        # Basic operations dictionary
        self.ops = nn.ModuleDict({
            'none': nn.Identity(),
            'nor_conv_1x1': nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(C, C, 1, padding=0),
                nn.BatchNorm2d(C)
            ),
            'nor_conv_3x3': nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(C, C, 3, padding=1),
                nn.BatchNorm2d(C)
            ),
            'skip_connect': nn.Identity(),
            'avg_pool_3x3': nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                nn.BatchNorm2d(C)
            ),
        })

        # Parse architecture string
        nodes = arch_str.split('+')
        self.connections = []
        for node in nodes:
            ops = node.strip('|').split('|')
            node_ops = []
            for op in ops:
                if op:
                    operation, connection = op.split('~')
                    node_ops.append((operation, int(connection)))
            self.connections.append(node_ops)

        # Input stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1),
            nn.BatchNorm2d(C)
        )

        # Output head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        
        states = [x]
        # Process each node
        for node_ops in self.connections:
            node_inputs = []
            for op_name, idx in node_ops:
                node_inputs.append(self.ops[op_name](states[idx]))
            # Combine inputs (sum in this case)
            if node_inputs:
                states.append(sum(node_inputs))
            
        return self.classifier(states[-1])




class KNAS(pl.LightningModule):
    def __init__(self, lr=10e-3):
        super().__init__()
        self.lr = lr
        generate_random_number = np.random.randint(0, 10)
        results = np.load('KNAS/predicted_networks.npy')

        self.model = NASNet(results[generate_random_number])
        self.criterion = nn.CrossEntropyLoss()

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
        
    
