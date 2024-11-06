import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class TransformerModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_heads=1,
                num_layers=2, lr=1e-3):
        super(TransformerModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc_out(x)

        performance = torch.sigmoid(x[:, 0])
        emissions = torch.sigmoid(x[:, 1])

        return torch.stack([performance, emissions], dim=1)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, targets)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def plot_loss_and_mse(train_losses, val_losses, train_mses, val_mses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot MSE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mses, label='Training MSE')
    plt.plot(epochs, val_mses, label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('loss_mse_plot.png')
