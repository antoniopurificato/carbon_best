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
from src.predictor.prepare_data_new import ArchitectureDataset



import json
import numpy as np
import warnings

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
    print(len(full_dataset.valid_data))

    # Assume network_dict.json is already loaded:
    print('extracting nasbench data...')
    '''with open('data.json', 'r') as file:
        network_data = json.load(file)'''
    with open('first_2_elements.json', 'r') as file:
        network_data = json.load(file)  # Load the dictionary'''


    print("First 2 elements saved to first_2_elements.json")

    outer_key = ('nb_0', 'cifar10', 100, 256, 0.2)

    # Add your new datapoint:
    full_dataset.add_new_datapoint(outer_key, network_data)
    print(f' new len_dataset: {len(full_dataset)}')
    outer_key = ('nb_0', 'cifar10', 100, 256, 0.125)

    print(full_dataset.valid_data.__getitem__(outer_key))
