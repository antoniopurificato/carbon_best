# """
# This module contains the TransformerTimeSeries and TransformerPredictor classes for time series forecasting.
# """

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import csv
import os
import sys
import time
import yaml

from src.predictor.transformer_models import *


class TransformerTimeSeries(nn.Module):
    """
    Transformer for time series data.
    Args:
        num_features: Number of input features
        seq_len: Length of the input sequence
        num_targets: Number of target values to predict
        d_model: Dimension of the model
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        dropout: Dropout rate
        dim_feedforward: Dimension of the feedforward network
    Returns:
        output: Predictions for each target value
    """

    def __init__(
        self,
        num_features,
        seq_len,
        num_targets,
        model_config: str = "src/configs/predictor_config.yaml",
    ):
        super(TransformerTimeSeries, self).__init__()
        self.model_config = model_config
        self.extract_info()
        self.seq_len = seq_len

        # Input embedding for fixed-size input
        self.input_embedding = nn.Linear(num_features, self.d_model)

        # Dropout
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # Positional encoding for the repeated sequence
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, self.d_model))

        # Transformer encoder
        if self.transformer_type == "standard":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_prob,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_encoder_layers
            )
        elif self.transformer_type == "informer":
            encoder_layer = InformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_prob,
            )
            self.transformer_encoder = InformerEncoder(
                encoder_layer, num_layers=self.num_encoder_layers
            )
        elif self.transformer_type == "performer":
            encoder_layer = PerformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_prob,
            )
            self.transformer_encoder = PerformerEncoder(
                encoder_layer, num_layers=self.num_encoder_layers
            )
        else:
            raise ValueError(
                f"Transformer architecture {self.transformer_type} unknown."
            )

        # Output layer for multi-target predictions
        self.output_layer_first = nn.Linear(self.d_model, 1)
        self.output_layer_rest = nn.Linear(self.d_model, 1)

    def extract_info(self):
        with open(self.model_config, "r") as config_file:
            config = yaml.safe_load(config_file)
        self.d_model = config["train_config"]["d_model"]
        self.nhead = config["train_config"]["nhead"]
        self.num_encoder_layers = config["train_config"]["num_encoder_layers"]
        self.dropout_prob = config["train_config"]["dropout"]
        self.dim_feedforward = config["train_config"]["dim_feedforward"]
        self.lr = config["train_config"]["lr"]
        self.transformer_type = config["train_config"]["transformer_type"]

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_embedding(x)

        # Add positional encoding
        x = x + self.positional_encoding.unsqueeze(0)

        if src_key_padding_mask is not None:
            src_key_padding_mask = (x == -1).all(dim=-1)
            src_key_padding_mask = src_key_padding_mask.to(torch.bool)

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Predictions
        first_prediction = self.output_layer_first(output)
        rest_prediction = self.output_layer_rest(output)

        # Combine predictions
        output = torch.cat([first_prediction, rest_prediction], dim=-1)

        return output


class TransformerPredictor(pl.LightningModule):
    """
    PyTorch Lightning module for the Transformer model.
    """

    def __init__(
        self,
        num_features,
        seq_len,
        num_targets,
        model_config: str = "src/configs/predictor_config.yaml",
    ):
        super(TransformerPredictor, self).__init__()
        self.model_config = model_config
        self.save_hyperparameters()
        self.extract_info()
        self.train_epoch_losses = []
        self.validation_epoch_losses = []
        self.epoch_times = []

        self.model = TransformerTimeSeries(
            num_features=num_features,
            seq_len=seq_len,
            num_targets=num_targets,
            model_config=model_config,
        )
        self.criterion = F.l1_loss
        self.best_val_loss = float("inf")

        # For DWA: maintain history of losses
        self.loss_acc_history = []
        self.loss_em_history = []

        # Track computing time
        self.epoch_start_time = None

    def extract_info(self):
        with open(self.model_config, "r") as config_file:
            config = yaml.safe_load(config_file)
        self.d_model = config["train_config"]["d_model"]
        self.nhead = config["train_config"]["nhead"]
        self.num_encoder_layers = config["train_config"]["num_encoder_layers"]
        self.dropout_prob = config["train_config"]["dropout"]
        self.dim_feedforward = config["train_config"]["dim_feedforward"]
        self.lr = config["train_config"]["lr"]

    def forward(self, x, src_key_padding_mask=None):
        return self.model(
            x, src_key_padding_mask=src_key_padding_mask
        )  # passing masks to the model

    def compute_dwa_weights(self):
        if (
            len(self.loss_acc_history) < 2
        ):  # Minimum 2 epochs required for computing DWA
            return 0.5, 0.5  # Equal weighting initially

        # Rates of change
        r_acc = self.loss_acc_history[-1] / self.loss_acc_history[-2]
        r_em = self.loss_em_history[-1] / self.loss_em_history[-2]

        # Normalize to sum to 1
        alpha_acc = r_acc / (r_acc + r_em)
        alpha_em = r_em / (r_acc + r_em)

        return alpha_acc, alpha_em

    def training_step(self, batch, batch_idx):
        _, inputs, labels = batch
        # Pass the mask to the model
        predictions = self(inputs)
        pred_acc = predictions[:, :, 0]
        pred_em = predictions[:, :, 1]
        label_acc = labels[:, :, 0]
        label_em = labels[:, :, 1]

        loss_acc = self.criterion(pred_acc, label_acc, reduction="none")
        loss_em = self.criterion(pred_em, label_em, reduction="none")

        label_padding_mask = (labels == -1).all(dim=-1).to(torch.bool)

        # Mask the losses
        loss_acc = loss_acc.masked_fill(
            label_padding_mask, 0
        )  # Set padded positions to 0
        loss_em = loss_em.masked_fill(label_padding_mask, 0)

        # Compute per-timestep mean (ignoring padding)
        loss_acc_per_timestep = loss_acc.sum(dim=0) / (~label_padding_mask).sum(
            dim=0
        ).clamp(min=1)
        loss_em_per_timestep = loss_em.sum(dim=0) / (~label_padding_mask).sum(
            dim=0
        ).clamp(min=1)

        # Get weights
        alpha_acc, alpha_em = self.compute_dwa_weights()

        comp_loss = alpha_acc * loss_acc_per_timestep + alpha_em * loss_em_per_timestep

        loss = comp_loss.mean()

        loss_acc_mean = loss_acc.mean()
        loss_em_mean = loss_em.mean()

        self.log(
            "train_acc_loss",
            loss_acc_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_em_loss", loss_em_mean, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("w_acc_train", alpha_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("w_em_train", alpha_em, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)  # Save the epoch time

        # Log the epoch time to the progress bar and logs
        self.log("epoch_time", epoch_time, on_epoch=True, prog_bar=True)

        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        train_acc_loss = self.trainer.callback_metrics.get("train_acc_loss", None)
        train_em_loss = self.trainer.callback_metrics.get("train_em_loss", None)
        w_acc_train = self.trainer.callback_metrics.get("w_acc_train", None)
        w_em_train = self.trainer.callback_metrics.get("w_em_train", None)
        self.train_epoch_losses.append(
            {
                "epoch": self.current_epoch,
                "train_loss": train_loss.item() if train_loss else None,
                "train_acc_loss": train_acc_loss.item() if train_acc_loss else None,
                "train_em_loss": train_em_loss.item() if train_em_loss else None,
                "w_acc_train": w_acc_train.item() if w_acc_train else None,
                "w_em_train": w_em_train.item() if w_em_train else None,
            }
        )
        self.loss_acc_history.append(train_acc_loss.item())
        self.loss_em_history.append(train_em_loss.item())

    def validation_step(self, batch, batch_idx):
        _, inputs, labels = batch
        predictions = self(inputs)
        pred_acc = predictions[:, :, 0]
        pred_em = predictions[:, :, 1]
        label_acc = labels[:, :, 0]
        label_em = labels[:, :, 1]

        loss_acc = self.criterion(pred_acc, label_acc, reduction="none")
        loss_em = self.criterion(pred_em, label_em, reduction="none")

        label_padding_mask = (labels == -1).all(dim=-1).to(torch.bool)

        loss_acc = loss_acc.masked_fill(label_padding_mask, 0)
        loss_em = loss_em.masked_fill(label_padding_mask, 0)

        loss_acc_per_timestep = loss_acc.sum(dim=0) / (~label_padding_mask).sum(
            dim=0
        ).clamp(min=1)
        loss_em_per_timestep = loss_em.sum(dim=0) / (~label_padding_mask).sum(
            dim=0
        ).clamp(min=1)

        alpha_acc, alpha_em = self.compute_dwa_weights()
        comp_loss = alpha_acc * loss_acc_per_timestep + alpha_em * loss_em_per_timestep

        loss = comp_loss.mean()
        loss_acc_mean = loss_acc.mean()
        loss_em_mean = loss_em.mean()

        self.log(
            "val_acc_loss", loss_acc_mean, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_em_loss", loss_em_mean, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("w_acc_val", alpha_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("w_em_val", alpha_em, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss", None)
        val_acc_loss = self.trainer.callback_metrics.get("val_acc_loss", None)
        val_em_loss = self.trainer.callback_metrics.get("val_em_loss", None)
        w_acc_val = self.trainer.callback_metrics.get("w_acc_val", None)
        w_em_val = self.trainer.callback_metrics.get("w_em_val", None)
        self.validation_epoch_losses.append(
            {
                "epoch": self.current_epoch,
                "val_loss": val_loss.item() if val_loss else None,
                "val_acc_loss": val_acc_loss.item() if val_acc_loss else None,
                "val_em_loss": val_em_loss.item() if val_em_loss else None,
                "w_acc_val": w_acc_val.item() if w_acc_val else None,
                "w_em_val": w_em_val.item() if w_em_val else None,
            }
        )

    def on_train_end(self):
        # File to save the losses
        file_path = "training_metrics.csv"

        # Define CSV header
        header = [
            "epoch",
            "val_loss",
            "train_loss",
            "val_acc_loss",
            "train_acc_loss",
            "val_em_loss",
            "train_em_loss",
            "w_acc_val",
            "w_acc_train",
            "w_em_val",
            "w_em_train",
            "epoch_time",
        ]

        # Write losses to CSV
        with open(os.path.join("results_csv", file_path), mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()  # Write the header

            # Write each epoch's losses
            for val_epoch_losses, train_epoch_losses, epoch_time in zip(
                self.validation_epoch_losses, self.train_epoch_losses, self.epoch_times
            ):
                writer.writerow(
                    {
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
                        "epoch_time": epoch_time,
                    }
                )

        print(f"Training and validation losses saved to {file_path}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=float(self.lr))
        return optimizer
