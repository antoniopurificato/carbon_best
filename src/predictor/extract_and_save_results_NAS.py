"""
Extract and Save Test Results 
This script handles the loading, processing, and evaluation of test datasets using a temporal transformer model.
It reads configurations from a YAML file, dynamically constructs test datasets and dataloaders, and processes
them to evaluate model performance and generate predictions. Results are saved as CSV files for further analysis.
"""

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import pickle
import torch
import numpy as np
import pandas as pd
import json
import os
import sys
import yaml
from datetime import date

from src.utils.main_utils import *
from src.predictor.temporal_transformer import TransformerPredictor
from src.predictor.prepare_data_NAS import ArchitectureDataset

#-------------------------------#
# Methods to process tests data 
#-------------------------------#
def process_test_dataloader(test_dataloader, labels_limit, test_name,seed, output_dir):
    """
    Process the given test dataloader and save the results to a CSV file.

    Args:
        test_dataloader (DataLoader): The PyTorch DataLoader containing the test dataset.
        labels_limit (int): The maximum number of labels to process per batch.
        test_name (str): The name of the test dataset being processed.
        seed (int): The random seed used for reproducibility and filename generation.
        output_dir (str): The directory where the result CSV will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the test predictions and true labels.
    """
    test_predictions, test_labels, test_exps, split_exps = [], [], [], []

    with torch.no_grad():
        for batch in test_dataloader:
            key, inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            print(f"NaNs in input: {torch.isnan(inputs).sum().item()}")
            print(f"Infs in input: {torch.isinf(inputs).sum().item()}")
            num_valid = (inputs != -1).sum().item()
            print(f"Valid input features for {key}: {num_valid} / {inputs.numel()}")

            key_str = "_".join(
                str(item[0]) if isinstance(item, tuple) else str(item.item()) if isinstance(item, torch.Tensor) else str(item)
                for item in key
            )

            test_exps.extend([key_str] * labels.size(1))

            key_parts = key_str.split("_")
            model_name, dataset_name, data_perc, bs, lr = parse_key_parts(key_parts)

            split_exps.extend([{"model_name": model_name, "dataset_name": dataset_name, "data_perc": data_perc, "bs": bs, "lr": lr}] * labels.size(1))

            #labels = labels[:labels_limit]
            #predictions = best_model(inputs)[:labels_limit]
            try:
                predictions = best_model(inputs)
                if predictions is None:
                    print(f"WARNING: No predictions returned for key {key_str}")
                    continue
                predictions = predictions[:labels_limit]
                labels = labels[:labels_limit]
            except Exception as e:
                print(f"ERROR running model on key {key_str}: {e}")
                continue

            test_predictions.append(predictions.cpu())
            test_labels.append(labels.cpu())

    test_predictions = torch.cat(test_predictions, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    test_data = prepare_results_dataframe(test_predictions, test_labels, split_exps, test_exps, labels_limit)

    if test_name == 'foursquare-tky':
        test_name = "foursquare_tky"

    csv_path = f"{test_name}_{seed}.csv"
    try:
        test_data.to_csv(os.path.join(f'src/{output_dir}', csv_path), index=False)
    except OSError:
        os.mkdir(f'src/{output_dir}')
        test_data.to_csv(os.path.join(f'src/{output_dir}', csv_path), index=False)
    print(f"Test results for {test_name} saved to {csv_path}")

    return test_data

def parse_key_parts(key_parts):
    """
    Parse the key parts from the test dataset keys to extract model and dataset configurations.

    Args:
        key_parts (list): A list of string components from the key.

    Returns:
        tuple: A tuple containing model name, dataset name, data percentage, batch size, and learning rate.
    """
    if len(key_parts) == 7:
        model_name = "_".join(key_parts[0:2])
        dataset_name = "_".join(key_parts[2:4])
        data_perc, bs, lr = key_parts[4:7]
    elif len(key_parts) == 6:
        model_name = key_parts[0]
        dataset_name = "_".join(key_parts[1:3])
        data_perc, bs, lr = key_parts[3:6]
    else:
        model_name = key_parts[0]
        dataset_name = key_parts[1]
        data_perc, bs, lr = key_parts[2:5]
    return model_name, dataset_name, data_perc, bs, lr

def prepare_results_dataframe(predictions, labels, split_exps, test_exps, labels_limit):
    """
    Prepare a DataFrame containing test predictions and true labels, along with experiment metadata.

    Args:
        predictions (torch.Tensor): The predicted outputs from the model.
        labels (torch.Tensor): The true labels from the dataset.
        split_exps (list): A list of dictionaries containing split experiment metadata.
        test_exps (list): A list of experiment keys for each test sample.
        labels_limit (int): The maximum number of labels to process per batch.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions, true labels, and metadata for each test sample.
    """
    pred_np = predictions.numpy()
    labels_np = labels.numpy()

    data = {
        "true_ACC": labels_np[:, :, 0].flatten(),
        "predicted_ACC": pred_np[:, :, 0].flatten(),
        "true_EN": labels_np[:, :, 1].flatten(),
        "predicted_EN": pred_np[:, :, 1].flatten(),
    }

    flattened_exps = pd.DataFrame(split_exps)
    original_key_strs = pd.DataFrame({"key_str": test_exps})

    df = pd.concat([original_key_strs.reset_index(drop=True), flattened_exps.reset_index(drop=True), pd.DataFrame(data)], axis=1)
    df['epoch'] = df.groupby('key_str').cumcount() + 1
    df= df[df['epoch'] <= labels_limit]

    valid_rows = (df['epoch'] <= labels_limit) & (df['true_ACC'] != -1) & (df['true_EN'] != -1)
    df = df[valid_rows]

    return df


if __name__ == '__main__':
    
    # Load datasets and models features 
    with open(load_yaml_exp_folder()[0], 'rb') as handle:
        datasets_to_features = pickle.load(handle)
    with open(load_yaml_exp_folder()[1], 'rb') as handle:
        models_to_features = pickle.load(handle)

    # Load test configurations
    with open("src/configs/predictor_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    test_names = config['test_config']['test_names']
    label_len = config['test_config']['label_len']
    output_dir = config['test_config']['output_dir']
    ckpt_dir= f"ckpt_NAS" #f"src/{date.today()}" #cambiare
    
    # General setting
    seed = config['seed']

    # Initialize the dataset and filter test data
    full_dataset = ArchitectureDataset(models_to_features, datasets_to_features)
    original_len=len(full_dataset.valid_data)
    print(f'len full_dataset: {original_len}')

    '''with open('data.json', 'r') as file:
        network_data = json.load(file)'''
    with open('first_2_elements.json', 'r') as file:
        network_data = json.load(file)

    # Iterate over network_data 
    for idx, data_entry in enumerate(network_data):  
        # Construct the outer_key with progressive numbering
        outer_key = (f'nb{idx}', 'cifar10', 100, 256, 0.2)
        full_dataset.add_new_datapoint(outer_key, network_data)

    print(f'Final dataset length: {len(full_dataset)}')
    print(f'Difference: {len(full_dataset)-original_len}, so added {int((len(full_dataset)-original_len)/4)} elements')
    test_data_all = full_dataset.filter_test_data(dataset_names=test_names)
    

    # Combine test datasets dynamically and exclude from training/validation
    all_test_data = {key: value for test_data in test_data_all for key, value in test_data.items()}
    print(f'test data all keys: {all_test_data.keys()}')
    remaining_data = {key: value for key, value in full_dataset.valid_data.items() if key not in all_test_data}
    full_dataset.valid_data = remaining_data

    # Dynamically create test datasets and dataloaders
    test_dataloaders = []
    for test_data in test_data_all:
        test_dataset = ArchitectureDataset(models_to_features, datasets_to_features)
        test_dataset.valid_data = test_data
        test_dataloaders.append(DataLoader(test_dataset, batch_size=1, shuffle=False))

    # Debug: Output sizes of test datasets
    for i, test_data in enumerate(test_data_all):
        print(f"Test Dataset {i + 1} ({test_names[i]}) Size:", len(test_data))

    # Efficiently collect maximum dimensions directly
    max_label_len = max_label_num = max_feat_num = 0
    for dataloader in test_dataloaders:
        for key, input, labels in dataloader:
            max_label_len = max(max_label_len, labels.shape[1])
            max_label_num = max(max_label_num, labels.shape[2])
            max_feat_num = max(max_feat_num, input.shape[1])

    print(f"Max label length: {max_label_len}, Max label num: {max_label_num}, Max feat num: {max_feat_num}")

    # Load model and prepare for evaluation
    num_features, seq_len, num_targets = max_feat_num, max_label_len, max_label_num
    checkpoint_path = os.path.join(ckpt_dir, f"{seed}.ckpt")
    try:
        best_model = TransformerPredictor.load_from_checkpoint(checkpoint_path, num_features=num_features, seq_len=seq_len, num_targets=num_targets)
    except FileNotFoundError:
        print("You did not train a model! I'll download a pre-trained checkpoint!")
        zip_name = f'src/{date.today()}.zip'
        download_data('1CGc9X7yJvRGB1eVYU7IvDr9ldYE0AkVq', zip_name)
        os.rename('src/ckpts_transformer', f'src/{date.today()}')
        best_model = TransformerPredictor.load_from_checkpoint(checkpoint_path, num_features=num_features, seq_len=seq_len, num_targets=num_targets)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model = best_model.to(device)
    best_model.eval()

    # Process each test dataloader 
    for dataloader, labels_limit, test_name in zip(test_dataloaders, label_len, test_names):
        process_test_dataloader(dataloader, labels_limit=labels_limit, test_name=test_name, seed=seed, output_dir=output_dir)
