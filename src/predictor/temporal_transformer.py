import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv
import os

class TransformerTimeSeries(nn.Module):
    def __init__(self, num_features, seq_len, num_targets, d_model=256, nhead=4, num_encoder_layers=4, dropout=0.1, kernel_size=3):
        super(TransformerTimeSeries, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Input embedding for fixed-size input
        self.input_embedding = nn.Linear(num_features, d_model)
        # Convolutional layers to extract local temporal patterns
        #self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size // 2)
        #self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size // 2)

        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding for the repeated sequence
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layer for multi-target predictions
        self.output_layer_first = nn.Linear(d_model, 1) 
        self.output_layer_rest = nn.Linear(d_model, 1)  
                
                

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_embedding(x)  # Shape: [batch_size, d_model]

        # Add positional encoding
        x = x + self.positional_encoding.unsqueeze(0)  # Shape: [batch_size, seq_len, d_model]
        
        if src_key_padding_mask is not None:
            src_key_padding_mask = (x == -1).all(dim=-1)  
            src_key_padding_mask = src_key_padding_mask.to(torch.bool)
        
        output = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        # Predictions
        first_prediction = self.output_layer_first(output)
        rest_prediction = self.output_layer_rest(output)

        # Combine predictions
        output = torch.cat([first_prediction, rest_prediction], dim=-1)

        return output

# Define the PyTorch Lightning Module
class TransformerPredictor(pl.LightningModule):
    def __init__(self, num_features, seq_len, num_targets, d_model=256, nhead=4, num_encoder_layers=4, lr=1e-2):
        super(TransformerPredictor, self).__init__()
        self.save_hyperparameters()
        self.train_epoch_losses = []
        self.validation_epoch_losses = []

        self.model = TransformerTimeSeries(
            num_features=num_features,
            seq_len=seq_len,
            num_targets=num_targets,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
        )
        self.criterion = F.l1_loss #.MSELoss() # use MAE because less sensitive to outliers
        self.best_val_loss = float('inf') 

        # For DWA: Maintain history of losses
        self.loss_acc_history = []
        self.loss_em_history = []

    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass, passing masks to the model.
        """
        return self.model(x, src_key_padding_mask=src_key_padding_mask)
    
    def compute_dwa_weights(self):
        if len(self.loss_acc_history) < 2: # Minimum 2 epochs required for computing DWA
            return 0.5, 0.5  # Equal weighting initially

        # rates of change
        r_acc = self.loss_acc_history[-1] / self.loss_acc_history[-2]
        r_em = self.loss_em_history[-1] / self.loss_em_history[-2]

        # normalize to sum to 1
        alpha_acc = r_acc / (r_acc + r_em)
        alpha_em = r_em / (r_acc + r_em)

        return alpha_acc, alpha_em
    '''
    # FOR MC Dropout 
    def predict_with_uncertainty(self, inputs, n_samples=50):
        self.model.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.model(inputs))
        self.model.eval()  # Disable dropout
        preds = torch.stack(preds)  # Shape: [n_samples, batch_size, seq_len, num_targets]
        mean_preds = preds.mean(dim=0)
        std_preds = preds.std(dim=0)
        return mean_preds, std_preds
    '''

    def training_step(self, batch, batch_idx):
        _, inputs, labels = batch
        # Pass the mask to the model
        predictions = self(inputs)
        #predictions = self(inputs, src_key_padding_mask=False)
        pred_acc= predictions[:, :, 0] 
        pred_em = predictions[:, :, 1] 
        label_acc = labels[:, :, 0]
        label_em = labels[:, :, 1]
        
        loss_acc = self.criterion(pred_acc, label_acc, reduction='none')  
        loss_em = self.criterion(pred_em, label_em, reduction='none')  

        label_padding_mask = (labels == -1).all(dim=-1).to(torch.bool)
   
        # Mask the losses
        loss_acc = loss_acc.masked_fill(label_padding_mask, 0)  # Set padded positions to 0
        loss_em = loss_em.masked_fill(label_padding_mask, 0)

        # Compute per-timestep mean (ignoring padding)
        loss_acc_per_timestep = loss_acc.sum(dim=0) / (~label_padding_mask).sum(dim=0).clamp(min=1)
        loss_em_per_timestep = loss_em.sum(dim=0) / (~label_padding_mask).sum(dim=0).clamp(min=1)

        # OLD: step-wise loss averaged over the batch
        #loss_acc_per_timestep = loss_acc.mean(dim=0)  # Shape: [seq_len]
        #loss_em_per_timestep = loss_em.mean(dim=0)  

        #get weights
        alpha_acc, alpha_em = self.compute_dwa_weights()

        comp_loss =  alpha_acc*loss_acc_per_timestep + alpha_em*loss_em_per_timestep
        #comp_loss =  0.7*loss_acc_per_timestep + 0.3*loss_em_per_timestep 
        
        loss = comp_loss.mean()

        loss_acc_mean = loss_acc.mean() #scalar
        loss_em_mean = loss_em.mean() #scalar

        self.log("train_acc_loss", loss_acc_mean, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_em_loss", loss_em_mean, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("w_acc_train", alpha_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("w_em_train", alpha_em, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        train_acc_loss = self.trainer.callback_metrics.get("train_acc_loss", None)
        train_em_loss = self.trainer.callback_metrics.get("train_em_loss", None)
        w_acc_train= self.trainer.callback_metrics.get("w_acc_train", None)
        w_em_train=self.trainer.callback_metrics.get("w_em_train", None)
        self.train_epoch_losses.append({
            "epoch": self.current_epoch,
            "train_loss": train_loss.item() if train_loss else None,
            "train_acc_loss": train_acc_loss.item() if train_acc_loss else None,
            "train_em_loss": train_em_loss.item() if train_em_loss else None,
            "w_acc_train": w_acc_train.item() if w_acc_train else None,
            "w_em_train": w_em_train.item() if w_em_train else None
         
        })
        self.loss_acc_history.append(train_acc_loss.item())
        self.loss_em_history.append(train_em_loss.item())

    def validation_step(self, batch, batch_idx):
        _, inputs, labels = batch
        predictions = self(inputs)
        pred_acc= predictions[:, :, 0] 
        pred_em = predictions[:, :, 1] 
        label_acc = labels[:, :, 0]
        label_em = labels[:, :, 1]

        #print(f"shape pred_acc {pred_acc.shape}" )
        #print(f"shape pred_em {pred_em.shape}" )
        #print(f"shape label_acc {label_acc.shape}" )
        #print(f"shape label_em {label_em.shape}" )

        loss_acc = self.criterion(pred_acc, label_acc, reduction='none')  
        loss_em = self.criterion(pred_em, label_em, reduction='none')  
        #print(f"loss_acc pre-mask shape: {loss_acc.shape}")
        #print(f"loss_em pre-mask  shape: {loss_em.shape}")


        label_padding_mask = (labels == -1).all(dim=-1).to(torch.bool)
        #print(f"label_padding_mask shape: {label_padding_mask}")

        # Mask the losses
        loss_acc = loss_acc.masked_fill(label_padding_mask, 0)  # Set padded positions to 0
        loss_em = loss_em.masked_fill(label_padding_mask, 0)
        #print(f"loss_acc shape: {loss_acc.shape}")
        #print(f"loss_em shape: {loss_em.shape}")


        # Compute per-timestep mean (ignoring padding)
        loss_acc_per_timestep = loss_acc.sum(dim=0) / (~label_padding_mask).sum(dim=0).clamp(min=1)
        loss_em_per_timestep = loss_em.sum(dim=0) / (~label_padding_mask).sum(dim=0).clamp(min=1)
        #print(f"loss_acc per timestep shape: {loss_acc_per_timestep.shape}")
        #print(f"loss_em per timestep shape: {loss_em_per_timestep.shape}")

       
        #OLD:
        #loss_acc_per_timestep = loss_acc.mean(dim=0)  
        #loss_em_per_timestep = loss_em.mean(dim=0)  


        alpha_acc, alpha_em = self.compute_dwa_weights()
        comp_loss =  alpha_acc*loss_acc_per_timestep + alpha_em*loss_em_per_timestep 
        #comp_loss =  0.7*loss_acc_per_timestep + 0.3*loss_em_per_timestep 
        
        
        loss = comp_loss.mean()
        loss_acc_mean = loss_acc.mean() 
        loss_em_mean = loss_em.mean() 

        # Compute baseline loss
        self.log("val_acc_loss", loss_acc_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_em_loss", loss_em_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("w_acc_val", alpha_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("w_em_val", alpha_em, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss", None)
        val_acc_loss = self.trainer.callback_metrics.get("val_acc_loss", None)
        val_em_loss = self.trainer.callback_metrics.get("val_em_loss", None)
        w_acc_val=self.trainer.callback_metrics.get("w_acc_val", None)
        w_em_val=self.trainer.callback_metrics.get("w_em_val", None)
        self.validation_epoch_losses.append({
            "epoch": self.current_epoch,
            "val_loss": val_loss.item() if val_loss else None,
            "val_acc_loss": val_acc_loss.item() if val_acc_loss else None,
            "val_em_loss": val_em_loss.item() if val_em_loss else None,
            "w_acc_val": w_acc_val.item() if w_acc_val else None,
            "w_em_val": w_em_val.item() if w_em_val else None
        })
    
    def on_train_end(self):
        # File to save the losses
        file_path = "training_validation_losses.csv"

        # Define CSV header
        header = [
            "epoch", "val_loss", "train_loss", 
            "val_acc_loss", "train_acc_loss", 
            "val_em_loss", "train_em_loss", 
            "w_acc_val", "w_acc_train", 
            "w_em_val", "w_em_train"
        ]

        # Write losses to CSV
        with open(os.path.join("results_csv", file_path), mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()  # Write the header

            # Write each epoch's losses
            for val_epoch_losses, train_epoch_losses in zip(self.validation_epoch_losses, self.train_epoch_losses):
                writer.writerow({
                    "epoch": val_epoch_losses["epoch"],
                    "val_loss": val_epoch_losses["val_loss"],
                    "train_loss": train_epoch_losses["train_loss"],
                    "val_acc_loss": val_epoch_losses["val_acc_loss"],
                    "train_acc_loss": train_epoch_losses["train_acc_loss"],
                    "val_em_loss": val_epoch_losses["val_em_loss"],
                    "train_em_loss": train_epoch_losses["train_em_loss"],
                    "w_acc_val": val_epoch_losses["w_acc_val"],
                    "w_acc_train": train_epoch_losses["w_acc_train"],
                    "w_em_val": val_epoch_losses["w_em_val"],
                    "w_em_train": train_epoch_losses["w_em_train"],
                })

        print(f"Training and validation losses saved to {file_path}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-3)
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    