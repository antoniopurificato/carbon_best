"""
ArchitectureDataset Module
==========================

This module defines the ArchitectureDataset class, which inherits from PyTorch's Dataset
and handles the loading, validation, and preprocessing of experimental data related to
model architectures, training metrics, hardware information, and more. 

Key Features:
-------------
- **Data Loading & Validation:**
  - Processes experimental folders, extracts relevant metrics (accuracy, energy consumption, etc.),
    and filters out invalid experiments.
- **Data Preprocessing & Normalization:**
  - Pads varying-length inputs and targets to fixed sizes.
  - Normalizes numerical fields (e.g., min-max scaling).
- **Utility for Training/Testing Splits:**
  - Provides methods to filter data by dataset names, compute statistics, and handle test datasets.

File Input/Output:
------------------
- **Input:** CSV files (metrics, emissions, etc.) organized in experiment-specific folders.
- **Output:** Preprocessed data in the form of PyTorch tensors, ready for use in a DataLoader.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import json
import warnings
import os
import sys

from src.utils.secondary_utils import get_models
from src.utils.main_utils import *

class ArchitectureDataset(Dataset):
    """
    This PyTorch Dataset which is responsible for:
    1) Loading and validating precomputed model experiment data. 
    2) Filtering out invalid experiments (e.g., missing or incorrect shapes).
    3) Padding and normalizing input features and target sequences for consistent tensor shapes.
    4) Providing data entries in a form suitable for PyTorch's DataLoader.
    """
    def __init__(self, model_dict: dict, data_dict: dict, is_dataset_new=False, max_len_labels=400):
        """
        Initializes the ArchitectureDataset and processes the provided data.

        Args:
            model_dict (dict): Dictionary of model-related metadata.
            data_dict (dict): Dictionary of dataset-related metadata.
            is_dataset_new (bool, optional): Whether to compute new min/max statistics 
                or load precomputed ones. Defaults to False.
            max_len_labels (int, optional): Maximum length for label sequences. Defaults to 400.
        """
        self.model_dict = model_dict
        self.data_dict = data_dict
        self.data_names = {'cifar10' : 'CIFAR10', 'cifar100' : 'CIFAR100',
        'mnist' : 'MNIST', 'food101' : 'food101', 'fashionmnist': 'FashionMNIST', 'google_boolq':'google_boolq' ,'stanfordnlp_imdb': 'stanfordnlp_imdb',
        'dair-ai_emotion' : 'dair-ai_emotion', 'ml-1m': 'ML-1M', 'foursquare-nyc': 'foursquare-nyc', 'amazon-beauty': 'amazon-beauty', 'foursquare-tky': 'foursquare-tky', 'rotten_tomatoes':'rotten_tomatoes'}
        self.model_names = {'vit' : 'vit_b_16', 'efficientnet' : 'efficientnet_b0',
        'squeezenet' : 'squeezenet1_0', 'resnet18' : 'resnet18',
        'alexnet' : 'alexnet', 'vgg16' : 'vgg16', 'bert-base-uncased': 'bert-base-uncased', 'roberta-base':'roberta-base',
        'microsoft_phi-2':'microsoft_phi-2', 'mistralai_Mistral-7B-v0.3':'mistralai_Mistral-7B-v0.3',
        'BERT4Rec': 'BERT4Rec' , 'CORE': 'CORE', 'GRU4Rec': 'GRU4Rec', 'SASRec': 'SASRec'}
        with open("src/configs/hardware_info.json", "r") as json_file: 
            self.hardware_mapping = json.load(json_file)
        if len(self.hardware_mapping) == 0:
            import cpuinfo

            properties = torch.cuda.get_device_properties(0)
            gpu_info = {key: getattr(properties, key) for key in dir(properties) if not key.startswith('_')}
            name = gpu_info['gcnArchName']
            self.hardware_mapping = {}
            self.hardware_mapping[name] = {'gpu' : gpu_info}
            del self.hardware_mapping[name]['gpu']['gcnArchName']
            self.hardware_mapping[name]['cpu'] = cpuinfo.get_cpu_info()


        self.model_dict = self.process_dict(model_dict)
        self.data_dict = self.process_dict(data_dict)
        self.entire_data = self.process_experiment_data()
        self.is_dataset_new=is_dataset_new
        self.max_len=max_len_labels

        self.valid_data = {}
        self.invalid_experiments = {}

        # Validate the shapes of the labels during initialization
        self.filter_valid_experiments()

        #Get max_padding and statistics to prepare input tensors
        if self.is_dataset_new==True: 
            self.max_padding = self.compute_max_padding()
            self.input_min=None
            self.input_max=None
            self.label_min=None
            self.label_max= None
            self.compute_statistics()
        else:
            with open("src/configs/max_padding_dict.json", "r") as json_file:
               self.max_padding = json.load(json_file)

            with open("src/configs/min_max_values_train.json", "r") as json_file:
                loaded_data = json.load(json_file)

            self.input_min = torch.tensor(loaded_data["input_min"])
            self.input_max = torch.tensor(loaded_data["input_max"])
            self.label_max= torch.tensor(loaded_data["label_max"])
            self.label_min= torch.tensor(loaded_data["label_min"])  

    def __len__(self):
        """
        Returns the number of valid experiments currently in the dataset.

        Returns:
            int: Count of valid data entries.
        """
        return len(self.valid_data)

    def __getitem__(self, idx):
            """
            Retrieves a single data item by index.

            This method:
            1) Extracts the relevant feature fields from the dataset and pads them.
            2) Gathers target fields (val_acc, energy_cumulative) into a label tensor.
            3) Normalizes both inputs and labels where applicable.

            Args:
                idx (int): The index of the dataset item to retrieve.

            Returns:
                tuple: (data_key, input_tensor, label_tensor_norm)
                    - data_key: The unique key identifying the experiment (e.g., (model_name, dataset_name, ...))
                    - input_tensor (torch.Tensor): The preprocessed and optionally normalized inputs.
                    - label_tensor_norm (torch.Tensor): The preprocessed and normalized label data.
            """
            key = list(self.valid_data.keys())[idx] 
            value = self.valid_data[key]

            # Extract target tensors
            target_keys = ['val_acc','energy_cumulative']
            labels = {k: value[k] for k in target_keys if k in value}
            label_tensors = []

            for k in target_keys:
                if k in labels:
                    sequence = labels[k]

                    if isinstance(sequence, (list, np.ndarray)):
                        # Handle the special cases 
                        if len(sequence)== 101 or len(sequence)==102:
                           sequence = sequence[:100] 
                        elif len(sequence)==7 or len(sequence)==6 :
                            sequence=sequence[:5]
                        sequence = sequence[:self.max_len]
                        # Pad the sequence to max_len
                        padded = np.pad(
                            sequence,
                            (0, max(0, self.max_len - len(sequence))),
                            constant_values=-1
                        )
                        label_tensors.append(torch.tensor(padded, dtype=torch.float32))
                    elif isinstance(sequence, torch.Tensor):
                        if sequence.size(0) == 101:
                            sequence = sequence[:100]  
                        # Truncate/pad to max_len
                        truncated = sequence[:self.max_len]
                        padded = torch.nn.functional.pad(
                            truncated,
                            (0, max(0, self.max_len - truncated.size(0))),
                            value=-1
                        )
                        label_tensors.append(padded.float())


            label_tensor = torch.stack(label_tensors, dim=-1) if label_tensors else torch.tensor([], dtype=torch.float32)
            mask = label_tensor != -1  # Shape: (seq_len, num_labels)

            label_tensor_norm = label_tensor.clone()

            if mask.any() and hasattr(self, 'label_min') and hasattr(self, 'label_max'):
                # Expand label_min and label_max for broadcasting along the sequence length
                label_min_expanded = self.label_min.view(1, -1)  # Shape: (1, num_labels)
                label_max_expanded = self.label_max.view(1, -1)  # Shape: (1, num_labels)

                # Use broadcasting to apply normalization element-wise where the mask is True
                label_tensor_norm[mask] = (
                    (label_tensor[mask] - label_min_expanded.expand_as(label_tensor)[mask]) /
                    (label_max_expanded.expand_as(label_tensor)[mask] - label_min_expanded.expand_as(label_tensor)[mask] + 1e-12)
                )

            # Extract and pad input features
            input_keys = [k for k in value.keys() if k not in target_keys]
            inputs = {k: value[k] for k in input_keys}

            # Perform padding
            padded_inputs = self.pad_single_sample(inputs)
            tensor_inputs = torch.tensor(padded_inputs, dtype=torch.float32)

            # Create a mask for non-padding values
            padding_mask = tensor_inputs != -1  # Exclude padding values
            range_mask = torch.zeros_like(tensor_inputs, dtype=torch.bool)
            range_mask[20:122] = True  #avoid normalizing class_distribution since already normalized

            # Combine both masks
            mask = padding_mask & ~range_mask
            if mask.any():
                tensor_inputs[mask] = (tensor_inputs[mask] - self.input_min[mask]) / (
                    self.input_max[mask] - self.input_min[mask] + 1e-8)
                
            return key, tensor_inputs, label_tensor_norm

    def filter_test_data(self, dataset_names: list):
        """
        Partitions the valid_data into subsets based on provided dataset names.

        Args:
            dataset_names (list): A list containing three dataset names to split into test_data_1, test_data_2, test_data_3.

        Returns:
            tuple: (test_data_1, test_data_2, test_data_3) each a dictionary of valid experiments keyed by their unique IDs.
        """
        test_data_1 = {}
        test_data_2 = {}
        test_data_3 = {}

        for key, value in self.valid_data.items():
            # Unpack the key to extract model_name and dataset_name
            model_name, data_name, discard, batch_size, learning_rate = key

            # Filter datasets based on the provided dataset names
            if data_name == dataset_names[0]:
                test_data_1[key] = value
            elif data_name == dataset_names[1]:
                test_data_2[key] = value
            elif data_name == dataset_names[2]:
                test_data_3[key] = value

        # Compute the remaining data by excluding the test datasets
        return test_data_1, test_data_2, test_data_3
    
    def compute_statistics(self):
        """
        Computes and saves min/max statistics for both inputs and labels across the valid data.

        - Collects all input fields, pads them, and computes elementwise min/max (ignoring padding = -1).
        - Collects and pads label fields (`val_acc`, `energy_cumulative`) and computes min/max ignoring padding.

        The statistics are saved to a JSON file 'min_max_values_train_new.json' for later reuse.
        """
        all_inputs = []
        target_keys = ['val_acc', 'energy_cumulative']

        for key, value in self.valid_data.items():
            input_keys = [k for k in value.keys() if k not in target_keys]
            inputs = {k: value[k] for k in input_keys}

            padded_sample = self.pad_single_sample(inputs)
            tensor_value = torch.tensor(padded_sample, dtype=torch.float32)
            all_inputs.append(tensor_value)

        all_inputs = torch.stack(all_inputs)     
        mask = all_inputs != -1  

        self.input_min = torch.min(all_inputs.masked_fill(~mask, float('inf')), dim=0).values
        self.input_max = torch.max(all_inputs.masked_fill(~mask, float('-inf')), dim=0).values

        label_masks = []  
        all_labels = []
        max_value_keys = {key: None for key in target_keys}  # Store the key for each max value
        max_values = {key: float('-inf') for key in target_keys}  # Initialize max values for each label

        # Iterate through valid_data and pad all labels to max length
        for key, value in self.valid_data.items():
            labels = []
            masks = []  # To store masks for each label field
            for k in target_keys:
                if k in value:   
                    sequence = value[k]
                    # Handle the special cases
                    if len(sequence) > self.max_len:
                        sequence = sequence[:self.max_len]
                    if len(sequence) ==101 or len(sequence)==102:
                        sequence = sequence[:100]
                    if len(sequence) ==7 or len(sequence)==6:
                        sequence = sequence[:5]

                    # Pad the sequence to max_len
                    padded = np.pad(
                        sequence,
                        (0, max(0, self.max_len - len(sequence))),
                        constant_values=-1
                    )
                    tensor_label = torch.tensor(padded, dtype=torch.float32)
                    labels.append(tensor_label)
                    masks.append(tensor_label != -1)  # Create a mask for non-padding values

                    # Check and update the maximum value and its key
                    max_seq_value = max(sequence)
                    if max_seq_value > max_values[k]:
                        max_values[k] = max_seq_value
                        max_value_keys[k] = key  

            if labels:
                stacked_labels = torch.stack(labels, dim=-1)  # Shape: (seq_len, num_targets)
                stacked_masks = torch.stack(masks, dim=-1)  # Corresponding mask
                all_labels.append(stacked_labels) 
                label_masks.append(stacked_masks)

        if all_labels:
            all_labels = torch.cat(all_labels, dim=0) 
            label_masks = torch.cat(label_masks, dim=0)  
            # Compute min and max values for non-padding elements
            self.label_min = torch.min(all_labels.masked_fill(~label_masks, float('inf')), dim=0).values
            self.label_max = torch.max(all_labels.masked_fill(~label_masks, float('-inf')), dim=0).values
        else:
            self.label_min = torch.tensor([])  # 
            self.label_max = torch.tensor([])

        data_to_save = {
            "input_min": self.input_min.tolist(),
            "input_max": self.input_max.tolist(), 
            "label_min": self.label_min.tolist(),
            "label_max": self.label_max.tolist()}

        # Save to JSON file
        with open("min_max_values_train_new.json", "w") as json_file:
            json.dump(data_to_save, json_file)

    def filter_valid_experiments(self):
        """
        Identifies and stores valid experiments in `self.valid_data` while logging invalid experiments in `self.invalid_experiments`.

        A valid experiment requires:
          1) 'val_acc' and 'energy_cumulative' have sufficient length based on model_name.
          2) No NaN/Inf values in those targets.

        Invalid experiments are stored in `self.invalid_experiments` with details about why they're invalid.
        """
        target_keys = ['val_acc','energy_cumulative'] 
        for key, value in self.entire_data.items():
           
            valid = True
            label_shapes = []

            # Determine the expected target length based on the model name
            model_name = key[0]  
            if model_name in ["bert-base-uncased", "roberta-base","microsoft_phi-2", "mistralai_Mistral-7B-v0.3"]:
                expected_length = 5
                check_condition = lambda x: len(x) >= expected_length
            elif model_name in ["BERT4Rec", "CORE", "GRU4Rec", "SASRec"]:
                expected_length = 400
                check_condition = lambda x: len(x) >= expected_length
            else:
                expected_length = 100
                check_condition = lambda x: len(x) >= expected_length

            # Check each target for the required shape and invalid values
            for target_key in target_keys:

                if target_key in value:
                    target = value[target_key]
                    # Check for NaNs, Infs, and target length
                    if isinstance(target, (list, np.ndarray)):
                        label_shapes.append(len(target))
                        if not check_condition(target) or np.isnan(target).any() or np.isinf(target).any():
                            valid = False
                    elif isinstance(target, torch.Tensor):
                        label_shapes.append(target.size(0))
                        if not check_condition(target) or torch.isnan(target).any() or torch.isinf(target).any():
                            valid = False

            # Exclude invalid experiments
            if valid:
                self.valid_data[key] = value
            else:
                self.invalid_experiments[key] = {
                    'label_shapes': label_shapes,
                    'invalid_fields': [
                        field for field in target_keys
                        if field in value and (
                            isinstance(value[field], (list, np.ndarray)) and 
                            (np.isnan(value[field]).any() or np.isinf(value[field]).any())
                            or isinstance(value[field], torch.Tensor) and 
                            (torch.isnan(value[field]).any() or torch.isinf(value[field]).any())
                        )
                    ]
                }

        for exp_name, details in self.invalid_experiments.items():
            warnings.warn(f"  Experiment: {exp_name}, Label shapes: {details['label_shapes']}, Invalid fields: {details.get('invalid_fields', [])}")
        print(f'Total number of invalid experiments: {len(self.invalid_experiments)}')

    def process_dict(self, single_feature):
        """
        Recursively processes nested structures in a dictionary or list, leaving scalars unchanged.

        Args:
            single_feature (any): Could be a dict, list, or a scalar type.

        Returns:
            same type as single_feature: 
                A processed copy with nested structures traversed and handled if necessary.
        """
        if isinstance(single_feature, dict):
            return {k: self.process_dict(v) for k, v in single_feature.items()}
        elif isinstance(single_feature, list):
            return [self.process_dict(v) for v in single_feature]
        else:
            return single_feature
        
    def pad_single_sample(self, data_item: dict)-> torch.Tensor:
        """
        Pads a single input sample's features based on self.max_padding.

        1) Iterates through each feature, flattens nested structures (lists, dicts, tuples) into a single list.
        2) Pads that list to the maximum required length found in self.max_padding for that field.
        3) Appends a one-hot encoding for activation functions, if present.

        Args:
            data_item (dict): Dictionary representing a single experiment's input features.

        Returns:
            torch.Tensor: A 1D tensor containing the flattened, padded features with -1 used for padding.
        """
        combined_padded_data = []
        for field, max_len in self.max_padding.items():
            if field in data_item:
                if isinstance(data_item[field], list):
                    field_data = data_item[field]
                    flattened_field_data = []

                    for sub_entry in field_data:
                        if isinstance(sub_entry, dict):
                            values = [v for k, v in sub_entry.items() if k != 'type']
                            for val in values:
                                if isinstance(val, tuple):
                                    flattened_field_data.extend(val)
                                else:
                                    flattened_field_data.append(val)
                        elif isinstance(sub_entry, tuple):
                            flattened_field_data.extend(sub_entry)
                        else:
                            flattened_field_data.append(sub_entry)
                    
                    if any(np.isnan(flattened_field_data)) or any(np.isinf(flattened_field_data)):
                        warnings.warn(f"Invalid raw data in field '{field}': {flattened_field_data}")
                        pass

                    # Pad to max length
                    padded_field = np.pad(
                        flattened_field_data[:max_len],
                        (0, max(0, max_len - len(flattened_field_data))),
                        constant_values=-1
                    )
                    combined_padded_data.extend(padded_field.tolist())
                elif isinstance(data_item[field], torch.Tensor):
                    tensor_data = data_item[field].cpu().numpy().tolist()
                    if any(np.isnan(tensor_data)) or any(np.isinf(tensor_data)):
                        warnings.warn(f"Invalid tensor data in field '{field}': {tensor_data}")
                        pass

                    padded_field = np.pad(
                        tensor_data[:max_len],
                        (0, max(0, max_len - len(tensor_data))),
                        constant_values=-1
                    )
                    combined_padded_data.extend(padded_field.tolist())
                else:
                    combined_padded_data.append(float(data_item[field]))

        # Append one-hot encoding for activation functions
        one_hot_activation = [0, 0, 0, 0] 
        if 'activation_functions_NEW' in data_item and data_item['activation_functions_NEW']:
            for activation in data_item['activation_functions_NEW']:
                if isinstance(activation, dict) and 'type' in activation:
                    if activation['type'] == 'ReLU':
                        one_hot_activation[0] = 1
                    elif activation['type'] == 'Sigmoid':
                        one_hot_activation[1] = 1
                    elif activation['type'] == 'SeLU':
                        one_hot_activation[2] = 1
                    elif activation['type'] == 'GELU':
                        one_hot_activation[3] = 1
        combined_padded_data.extend(one_hot_activation)

        return torch.tensor(combined_padded_data, dtype=torch.float32)


    def compute_max_padding(self):
        """
        Computes the maximum length for each feature key across all valid experiments.

        Scans through self.valid_data to determine the largest list/tuple length per feature field,
        which is then used for padding. For scalar fields, the max length is 1.

        Returns:
            dict: {field_name: max_length, ...}
        """
        max_padding = {}
        for _, data_item in self.valid_data.items():
            for key, value in data_item.items():
                if key == 'activation_functions_NEW':
                    continue

                if isinstance(value, list):
                    total_length = 0
                    for sub_entry in value:
                        if isinstance(sub_entry, dict):
                            stripped_values = [v for k, v in sub_entry.items() if k != 'type']
                            for val in stripped_values:
                                if isinstance(val, tuple):
                                    total_length += len(val)
                                elif isinstance(val, list):
                                    total_length += len(val)
                                else:
                                    total_length += 1
                        elif isinstance(sub_entry, tuple):
                            total_length += len(sub_entry)
                        else:
                            total_length += 1

                    max_padding[key] = max(max_padding.get(key, 0), total_length)

                elif isinstance(value, (np.ndarray, torch.Tensor)):
                    max_padding[key] = max(max_padding.get(key, 0), len(value))

                elif isinstance(value, (int, float, str)):
                    max_padding[key] = 1  # Scalars occupy a single position

        return max_padding


    def obtain_precomputed_results(self, model_name:str, dataset_name:str,
                                   exp_name:str):
        """
        Fetches precomputed metrics, emissions, and hardware info for a specific experiment.

        Args:
            model_name (str): The name of the model.
            dataset_name (str): The dataset used (e.g., 'cifar10').
            exp_name (str): The constructed experiment folder name (e.g., 'discard_30_0.001').

        Returns:
            tuple: (metrics_tensor, emissions_tensor, energy_tensor, gpu_info, cpu_info, ram_total_size)
        """
        
        NOT_FOUND = False
        results_folder = Path(load_yaml_exp_folder()[2])                                                                                   
        folder_name = os.path.join(results_folder, model_name, f'{dataset_name.lower()}_{exp_name}') 
        folder_name = results_folder / model_name / f"{dataset_name.lower()}_{exp_name}"

        try:
            if model_name in ["bert-base-uncased", "roberta-base","microsoft_phi-2", "mistralai_Mistral-7B-v0.3"]:
                metrics_df = pd.read_csv(f'{folder_name}/epochs_results/training_log.csv')
                val_accuracy_df = metrics_df[metrics_df['Metric_Name'] == 'val_accuracy']
                metrics = val_accuracy_df['Value'].dropna().values.tolist()
            else:
                if model_name in ["BERT4Rec", "CORE", "GRU4Rec", "SASRec"]:
                    try:
                        metrics = pd.read_csv(f'{folder_name}/version_0/metrics.csv')['val_NDCG_@10/dataloader_idx_0'].dropna().values.tolist()   
                    except FileNotFoundError:
                        metrics = pd.read_csv(f'{folder_name}/version_1/metrics.csv')['val_NDCG_@10/dataloader_idx_0'].dropna().values.tolist()
                else:        
                    try:
                        metrics = pd.read_csv(f'{folder_name}/version_0/metrics.csv')['val_acc'].dropna().values.tolist()
                    except FileNotFoundError:
                        metrics = pd.read_csv(f'{folder_name}/version_1/metrics.csv')['val_acc'].dropna().values.tolist()


        except FileNotFoundError:
                NOT_FOUND = True
                warnings.warn(f'No metrics found for {model_name} on {dataset_name} with {exp_name}')

        if not NOT_FOUND:
            if model_name in  ["BERT4Rec", "CORE", "GRU4Rec", "SASRec"]:
                emissions_data = pd.read_csv(f'{folder_name}/emissions_epochs.csv')
            else:
                emissions_data = pd.read_csv(f'{folder_name}/emissions.csv')
            emissions_codecarbon = emissions_data['emissions'].values.tolist()
            energy=emissions_data['energy_consumed'].values.tolist()
            gpu_model=emissions_data['gpu_model'][0][4:]
            ram_total_size = emissions_data['ram_total_size'][0]

            gpu_keys = ["major", "minor", "total_memory", "multi_processor_count", "L2_cache_size"]
            cpu_keys = ["bits", "count", "hz_advertised"]

            if gpu_model == None:
                gpu_info = {key: -1 for key in gpu_keys}
            gpu_info = extract_hardware_info_from_file(self.hardware_mapping[gpu_model]['gpu'], gpu_keys)
    
            cpu_info = extract_hardware_info_from_file(self.hardware_mapping[gpu_model]['cpu'], cpu_keys)
            if cpu_info and "hz_advertised" in cpu_info:
                cpu_info["hz_advertised"] = cpu_info["hz_advertised"][0]  
            

            return (
                torch.tensor(metrics),
                torch.tensor(emissions_codecarbon),
                torch.tensor(energy),
                gpu_info,
                cpu_info,
                ram_total_size
            )

    def process_experiment_data(self) -> dict:
        """
        Iterates through a results folder structure, parsing experiment metadata 
        for all recognized models and collecting relevant performance metrics 
        (accuracy, energy consumption, hardware info, etc.).

        Returns:
            dict: 
                A dictionary where each key is a tuple:
                    (model_name, dataset_name, discard_percentage, batch_size, learning_rate)
                and each value is a dictionary of processed metrics and metadata.
        """
        final_data = {}
        all_models = get_models()  
        results_folder = Path(load_yaml_exp_folder()[2])  
        for model_folder in results_folder.iterdir():
            if 'mistralai' in str(model_folder) or 'microsoft' in str(model_folder):
                new_path = str(model_folder).replace('_', '/')
                model_folder = Path(new_path)
                if not is_valid_model_folder(model_folder, all_models):
                    continue

            model_name = model_folder.name
            for exp_folder in model_folder.glob('*'):
                print(exp_folder)
                if not process_exp_folder(exp_folder):
                    continue

                # For each valid experiment folder, parse and collect data
                data_key = None
                try:
                    meta = parse_experiment_metadata(exp_folder, model_name)
                    if not meta:  # If parsing failed or folder format is invalid
                        continue

                    # Unique key for storing results
                    data_key = (
                        model_name, 
                        meta["dataset"], 
                        meta["discard_pct"], 
                        meta["batch_size"], 
                        meta["lr"]
                    )

                    # Collect hardware/performance metrics
                    metrics = self._collect_metrics(exp_folder, meta)
                    final_data[data_key] = metrics

                    # Add dataset-level and other supplemental info
                    self._add_supplemental_data(final_data, exp_folder, data_key, meta)

                except Exception as e:
                    # If anything goes wrong, handle cleanup
                    handle_processing_error(exp_folder, e, final_data, data_key)

        return final_data

 
    def _collect_metrics(self, exp_folder: Path, meta: dict) -> dict:
        """
        Extracts model architecture metrics, energy usage, and performance results.

        Args:
            exp_folder (Path): Path to the experiment folder.
            meta (dict): Parsed metadata with model_name, dataset, discard_pct, etc.

        Returns:
            dict: Dictionary of collected metrics (FLOPS, depth, params, val_acc, 
                  energy_cumulative, etc.).
        """
        # Architecture metrics
        depth, params = (-1, -1)
        if exp_folder.exists():
            depth, params = extract_architecture_metrics(exp_folder)

        # Base metrics
        metrics = {
            "FLOPS": extract_flops_from_text(exp_folder) if exp_folder.exists() else -1,
            "depth": depth,
            "params": params,
            "discard_percentage": meta["discard_pct_comp"],
            "batch_size": meta["batch_size"],
            "learning_rate": meta["lr"],
        }

        # Collect precomputed results
        exp_name = construct_experiment_name(meta)
        results, emissions, energy, gpu_info, cpu_info, ram = self.obtain_precomputed_results(
            model_name=meta["model_name"],
            dataset_name=meta["dataset"],
            exp_name=exp_name
        )

        # Potentially shortened / processed energy array
        processed_energy = process_energy_data(energy, meta["model_name"])
        metrics.update({
            "val_acc": results,
            "energy_cumulative": torch.cumsum(processed_energy, dim=0),
            "ram_total_size": ram,
            "gpu_info": self._parse_gpu_info(gpu_info),
            "cpu_info": self._parse_cpu_info(cpu_info)
        })

        return metrics

    def _add_supplemental_data(self, final_data: dict, exp_folder: Path, key: tuple, meta: dict) -> None:
        """
        Adds additional dataset-related info, layer information, and cleans up unused fields.

        Args:
            final_data (dict): The overall dictionary storing processed experiment data.
            exp_folder (Path): The experiment folder path.
            key (tuple): Unique key for this experiment entry in final_data.
            meta (dict): Parsed metadata.
        """
        # Attach dataset info
        dataset_id = self.data_names.get(meta["dataset"])
        if dataset_id is None:
            raise KeyError(f"Dataset name {meta['dataset']} not recognized in self.data_names.")
        final_data[key].update(self.data_dict[dataset_id])

        # Calculate example counts
        calculate_example_counts(final_data, key, meta)

        # Parse and add layer info
        if exp_folder.exists():
            layer_info = parse_layers_info(exp_folder)
            final_data[key].update(layer_info)

        # Remove unused fields if they exist
        for field in [
            "pooling_layers", "emissions_codecarbon", "energy", 
            "num_test_examples", "task"
        ]:
            final_data[key].pop(field, None)

    def _parse_gpu_info(self, gpu_info: dict):
        """
        Parses GPU information and returns a list of GPU attributes.

        The expected keys in `gpu_info` are:
            "major", "minor", "total_memory", "multi_processor_count", "L2_cache_size"

        Args:
            gpu_info (dict): A dictionary containing GPU-related fields. 
                            Example:
                            {
                                "major": 7,
                                "minor": 5,
                                "total_memory": 16160,
                                "multi_processor_count": 48,
                                "L2_cache_size": 6144
                            }

        Returns:
            list or None: 
                A list of GPU attributes in the order [major, minor, total_memory, multi_processor_count, L2_cache_size].
                Returns None if `gpu_info` is empty.
        """
        if not gpu_info:
            return None
        return [
            gpu_info.get("major"),
            gpu_info.get("minor"),
            gpu_info.get("total_memory"),
            gpu_info.get("multi_processor_count"),
            gpu_info.get("L2_cache_size")
        ]


    def _parse_cpu_info(self, cpu_info: dict):
        """
        Parses CPU information and returns a list of CPU attributes.

        The expected keys in `cpu_info` are:
            "bits", "count", "hz_advertised"

        Args:
            cpu_info (dict): A dictionary containing CPU-related fields.
                            Example:
                            {
                                "bits": 64,
                                "count": 16,
                                "hz_advertised": "3.70GHz"
                            }

        Returns:
            list or None: 
                A list of CPU attributes in the order [bits, count, hz_advertised].
                Returns None if `cpu_info` is empty.
        """
        if not cpu_info:
            return None
        return [
            cpu_info.get("bits"),
            cpu_info.get("count"),
            cpu_info.get("hz_advertised")
        ]

