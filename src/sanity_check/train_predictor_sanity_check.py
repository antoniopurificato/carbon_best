"""
This script trains the TransformerPredictor model on the full dataset
(excluding specific test datasets: cifar10, rotten_tomatoes, foursquare-tky),
and saves the best-performing model checkpoint based on validation loss.
"""

import numpy as np
import pickle
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split, ConcatDataset
import os
import sys
import time
import yaml

from src.utils.secondary_utils import seed_everything
from src.utils.main_utils import load_yaml_exp_folder
from src.predictor.temporal_transformer_old import TransformerPredictor
from src.predictor.prepare_data import ArchitectureDataset


if __name__ == "__main__":
    # Load training configuration
    with open("src/configs/predictor_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    seed = config["seed"]
    seed_everything(seed)  # Set seed for reproducibility

    start = time.time()

    # Load preprocessed features
    exp_folder_paths = load_yaml_exp_folder()
    with open(exp_folder_paths[0], "rb") as handle:
        datasets_to_features = pickle.load(handle)
    with open(exp_folder_paths[1], "rb") as handle:
        models_to_features = pickle.load(handle)

    # Initialize the full dataset from loaded feature dictionaries
    full_dataset = ArchitectureDataset(models_to_features, datasets_to_features)

    # Filter out test datasets from the full dataset
    cv_data, nlp_data, recommendation_data = full_dataset.filter_test_data(
        dataset_names=["cifar10", "rotten_tomatoes", "foursquare-tky"]
    )

    # Merge test dataset entries into a single exclusion dictionary
    all_test_data = {**cv_data, **nlp_data, **recommendation_data}

    # Remove test entries from the dataset's valid_data
    full_dataset.valid_data = {
        key: value
        for key, value in full_dataset.valid_data.items()
        if key not in all_test_data
    }

    # Split remaining data into training and validation sets (80/20 split)
    val_split_ratio = 0.20
    train_size = int(len(full_dataset) * (1 - val_split_ratio))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Double the training set to simulate a larger training size
    train_dataset = ConcatDataset([train_dataset, train_dataset])

    # Create PyTorch DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model input/output dimensions
    num_features = 1670
    seq_len = 400
    num_targets = 2

    # ModelCheckpoint callback to save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="ckpt",
        filename=f"{str(seed)}_sanity_check",
        save_top_k=1,
        mode="min",
        verbose=True,
    )

    # EarlyStopping callback to stop training if no improvement
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=12, verbose=True, mode="min"
    )

    # Initialize Transformer model
    model = TransformerPredictor(
        num_features=num_features, seq_len=seq_len, num_targets=num_targets
    )

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=32,
        log_every_n_steps=10,
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=True,
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    end = time.time()
    print(f"Time elapsed for training: {end - start:.2f} seconds")
