"""
This script is used to perform hyperparameter optimization for the transformer model.
"""

import wandb
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split
import sys
import os
import argparse
import yaml

from src.utils.main_utils import *
from src.predictor.temporal_transformer import TransformerPredictor
from src.predictor.prepare_data import ArchitectureDataset


PROJECT_NAME = 'None'

import yaml

import yaml

def update_train_config(config_dict, config_file_path='src/configs/predictor_config.yaml'):
    # Load existing config
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Ensure 'train_config' exists in the config
    if 'train_config' not in config:
        config['train_config'] = {}

    # Update only the specified fields in 'train_config'
    train_config = config['train_config']
    train_config['d_model'] = config_dict.d_model
    train_config['nhead'] = config_dict.num_heads
    train_config['num_encoder_layers'] = config_dict.layers
    train_config['dim_feedforward'] = config_dict.dim_feedforward
    train_config['lr'] = config_dict.lr

    # Write the updated config back to the file
    with open(config_file_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


def train_model_with_hpo(config=None):
    """
    Train the model with hyperparameter optimization. 
    """


    wandb.init(
        project=PROJECT_NAME,
        config=config,  
    )

    config = wandb.config
    learning_rate = config.lr
    num_heads = config.num_heads
    num_layers = config.layers
    d_model = config.d_model
    dim_feedforward = config.dim_feedforward
    transformer_type = config.transformer_type

    with open(load_yaml_exp_folder()[0], 'rb') as handle:
        datasets_to_features = pickle.load(handle)
    with open(load_yaml_exp_folder()[1], 'rb') as handle:
        models_to_features = pickle.load(handle)


    wandb_logger = WandbLogger(project=PROJECT_NAME, log_model=True)

    # Prepare dataloaders and take out test data
    full_dataset = ArchitectureDataset(models_to_features, datasets_to_features)
    cv_data, nlp_data, recommendation_data = full_dataset.filter_test_data(
        dataset_names=['cifar10', 'rotten_tomatoes', 'foursquare-tky']
    )
    all_test_data = {**cv_data, **nlp_data, **recommendation_data}
    remaining_data = {key: value for key, value in full_dataset.valid_data.items() if key not in all_test_data}
    full_dataset.valid_data = remaining_data

    val_split_ratio = 0.20
    train_size = int(len(full_dataset) * (1 - val_split_ratio))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print('')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath = f"ckpt/hpo/{learning_rate}_{num_heads}_{num_layers}_{d_model}_{dim_feedforward}_{transformer_type}/", filename="best_model", save_top_k=1, mode="min", verbose=True
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=15, mode="min", verbose=True
    )

    update_train_config(config)

    model = TransformerPredictor(
        num_features=1670, seq_len=400, num_targets=2
    )

    trainer= pl.Trainer(max_epochs=300, log_every_n_steps=10, callbacks=[early_stopping_callback, checkpoint_callback],
                        logger=wandb_logger, devices=1)

    trainer.fit(model, train_dataloader, val_dataloader)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_key", type=str, help="Wandb API key", required=True)
    parser.add_argument("--wandb_project", type=str, help="Wandb project name", required=True)
    args = parser.parse_args()
    wandb.login(key=args.wandb_key)

    PROJECT_NAME = args.wandb_project

    sweep_config = {
        "method": "bayes",  
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            #"lr": {"values": [0.001, 0.005, 0.0003]},
            "lr": {"values": [0.005]},
            #"num_heads": {"values": [4, 8]},
            "num_heads": {"values": [4]},
            #"layers": {"values": [4, 8, 12]},
            "layers": {"values": [4]},
            #"d_model": {"values": [256, 512]},
            "d_model": {"values": [256]},
            #"dim_feedforward": {"values": [512, 1024]},
            "dim_feedforward": {"values": [512]},
            "transformer_type": {"values": ["performer", "informer", "standard"]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=train_model_with_hpo, count=30)
