import torch
import torch.nn as nn
import pytorch_lightning as pl



class TransformerModel(pl.LightningModule):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int, 
                 num_heads: int = 1, 
                 num_layers: int = 2):
        """
        Initializes the TransformerModel.

        Parameters:
        -----------
        input_size : int
            Input feature size.
        hidden_size : int
            Hidden layer size.
        output_size : int
            Output feature size.
        num_heads : int, optional
            Number of attention heads (default is 1).
        num_layers : int, optional
            Number of transformer encoder layers (default is 2).
        """
        super(TransformerModel, self).__init__()

        # Transformer Encoder Layer with attention heads
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Fully connected layers
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Dropout and ReLU activation
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters:
        -----------
        x : Tensor
            Input tensor with shape (sequence_length, batch_size, input_size).

        Returns:
        --------
        Tensor:
            Output tensor with shape (batch_size, 2), where the first value represents
            performance and the second value represents emissions.
        """
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = self.relu(x)  # added by Pippo
        x = self.dropout(x)
        x = self.fc_out(x)

        performance = torch.sigmoid(x[:, 0])
        emissions = torch.sigmoid(x[:, 1])

        return torch.stack([performance, emissions], dim=1)

     def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Parameters:
        -----------
        batch : tuple
            A tuple containing (input_tensor, target_tensor).
        batch_idx : int
            Index of the batch.

        Returns:
        --------
        Tensor:
            The loss value for the current batch.
        """
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = F.binary_cross_entropy(y_hat, y)  # change according to ViT
        
        # Calculate MSE
        mse_train = F.mse_loss(y_hat, y)

        # Log the loss and MSE
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse_train, on_step=True, on_epoch=True)

        # Append losses to lists for tracking
        self.train_losses.append(loss.item())
        self.train_mses.append(mse_train.item())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Parameters:
        -----------
        batch : tuple
            A tuple containing (input_tensor, target_tensor).
        batch_idx : int
            Index of the batch.

        Returns:
        --------
        None
        """
        x, y = batch
        y_hat = self(x)

        # Calculate loss
        loss = F.binary_cross_entropy(y_hat, y)

        # Calculate MSE
        mse_val = F.mse_loss(y_hat, y)

        # Log the validation loss and MSE
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse_val, on_step=True, on_epoch=True)

        # Append losses to lists for tracking
        self.val_losses.append(loss.item())
        self.val_mses.append(mse_val.item())

        # Early stopping
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.save_best_model()

    def save_best_model(self):
        """
        Save the best model state.
        """
        torch.save(self.state_dict(), 'best_model.pth')
        print("Best model saved.")

    def configure_optimizers(self):
        """
        Configures the optimizer.

        Returns:
        --------
        Optimizer:
            The optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
