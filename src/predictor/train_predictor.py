"""
This script trains the TransformerPredictor model on the full dataset (excluding test datasets) and saves the best model.
"""

import numpy as np
import pickle
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split
import os
import sys
import time

from src.utils.secondary_utils import seed_everything
from src.utils.main_utils import *
from src.predictor.temporal_transformer import TransformerPredictor
from src.predictor.prepare_data import ArchitectureDataset


if __name__ == '__main__':
    
    with open("src/configs/predictor_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    seed = config['seed']
    seed_everything(seed)

    start = time.time()

    # Load the datasets and models dictionaries  
    try:
        with open(load_yaml_exp_folder()[0], 'rb') as handle:
            datasets_to_features = pickle.load(handle)
        with open(load_yaml_exp_folder()[1], 'rb') as handle:
            models_to_features = pickle.load(handle)
    except FileNotFoundError:
        print("You do not have the data! I'll download them")
        download_data()
        with open(load_yaml_exp_folder()[0], 'rb') as handle:
            datasets_to_features = pickle.load(handle)
        with open(load_yaml_exp_folder()[1], 'rb') as handle:
            models_to_features = pickle.load(handle)

    # Initialize the full dataset
    full_dataset = ArchitectureDataset(models_to_features, datasets_to_features)

    # Filter out the test datasets
    cv_data, nlp_data, recommendation_data = full_dataset.filter_test_data(
        dataset_names=['cifar10', 'rotten_tomatoes', 'foursquare-tky']
    )

    # Combine all test datasets into a single dictionary for exclusion
    all_test_data = {**cv_data, **nlp_data, **recommendation_data}

    # Exclude all test dataset keys from the training/validation set
    remaining_data = {key: value for key, value in full_dataset.valid_data.items() if key not in all_test_data}
    full_dataset.valid_data = remaining_data
    
    # Split the full dataset into training and validation sets
    val_split_ratio = 0.20 # Proportion of data for validation
    train_size = int(len(full_dataset) * (1 - val_split_ratio))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    num_features = 1670 
    seq_len = 400
    num_targets = 2

    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",           
    dirpath="ckpt",       
    filename=str(seed),  
    save_top_k=1,                
    mode="min", 
    verbose=True
)

    # Define the EarlyStopping callback
    early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # Metric to monitor
    patience=12,          # Number of epochs with no improvement after which training will stop
    verbose=True,        # Print early stopping message
    mode="min"           # "min" because we want to minimize the validation loss
)
    # Initialize the model
    model = TransformerPredictor(num_features=num_features, seq_len=seq_len, num_targets=num_targets)

    # Initialize the trainer
    trainer= pl.Trainer(max_epochs=300, log_every_n_steps=10, callbacks=[early_stopping_callback, checkpoint_callback], logger=True)
    # Train the model
    trainer.fit(model,train_dataloader, val_dataloader)

    end = time.time()

    print(f"Time elapsed for training: {end-start}")
