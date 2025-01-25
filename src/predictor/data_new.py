import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import pandas as pd
import os
from pathlib import Path
import numpy as np

from src.utils.secondary_utils import get_models
from src.utils.main_utils_new import *

class ArchitectureDataset(Dataset):
    def __init__(self, model_dict: dict, data_dict: dict):
        self.model_dict = model_dict
        self.data_dict = data_dict
        self.data_names = {'cifar10' : 'CIFAR10', 'cifar100' : 'CIFAR100',
        'mnist' : 'MNIST', 'food101' : 'food101', 'FashionMNIST': 'FashionMNIST', 'google_boolq':'google_boolq' ,'stanfordnlp_imdb': 'stanfordnlp_imdb',
        'dair-ai_emotion' : 'dair-ai_emotion', 'ml-1m': 'ML-1M'}
        self.model_names = {'vit' : 'vit_b_16', 'efficientnet' : 'efficientnet_b0',
        'squeezenet' : 'squeezenet1_0', 'resnet18' : 'resnet18',
        'alexnet' : 'alexnet', 'vgg16' : 'vgg16', 'bert-base-uncased': 'bert-base-uncased', 'roberta-base':'roberta-base',
        'microsoft_phi-2':'microsoft_phi-2', 'mistralai_Mistral-7B-v0.3':'mistralai_Mistral-7B-v0.3',
        'BERT4Rec': 'BERT4Rec' , 'CORE': 'CORE', 'GRU4Rec': 'GRU4Rec', 'SASRec': 'SASRec'}
        self.model_dict = self.process_dict(model_dict)
        self.data_dict = self.process_dict(data_dict)
        self.entire_data = self.obtain_entire_dataset()

        self.valid_data = {}
        self.invalid_experiments = {}

        # Validate the shapes of the labels during initialization
        self.filter_valid_experiments()
        self.max_padding = self.compute_max_padding()
        for k, v in self.max_padding.items():
            print(f'max padd :{k}, {v} ')
        #self.get_exp_with_max_conv_layers()
        self.label_max=None
        self.label_min=None
        self.input_min=None
        self.input_max=None
        self.compute_statistics()

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
            key = list(self.valid_data.keys())[idx] 
            value = self.valid_data[key]

            # Extract target tensors
            target_keys = ['val_acc','energy_cumulative']
            labels = {k: value[k] for k in target_keys if k in value}
            label_tensors = []
            max_len = 400  # Maximum length for truncation and padding

            for k in target_keys:
                if k in labels:
                    sequence = labels[k]

                    if isinstance(sequence, (list, np.ndarray)):
                        # Handle the special case where sequence length is 101
                        if len(sequence) == 101:
                            sequence = sequence[:100]  # Truncate to 100

                        # Truncate the sequence to max_len (400)
                        sequence = sequence[:max_len]

                        # Pad the sequence to max_len
                        padded = np.pad(
                            sequence,
                            (0, max(0, max_len - len(sequence))),
                            constant_values=-1
                        )
                        label_tensors.append(torch.tensor(padded, dtype=torch.float32))
                    elif isinstance(sequence, torch.Tensor):
                        # Handle the special case where sequence length is 101
                        if sequence.size(0) == 101:
                            sequence = sequence[:100]  # Truncate to 100

                        # Truncate/pad to max_len
                        truncated = sequence[:max_len]
                        padded = torch.nn.functional.pad(
                            truncated,
                            (0, max(0, max_len - truncated.size(0))),
                            value=-1
                        )
                        label_tensors.append(padded.float())

            # Stack label tensors
            label_tensor = torch.stack(label_tensors, dim=-1) if label_tensors else torch.tensor([], dtype=torch.float32)
            mask = label_tensor != -1  # Shape: (seq_len, num_labels)

            # Clone the label tensor to preserve original values
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
            padded_inputs = self.pad_single_sample(inputs)
        
            tensor_inputs = torch.tensor(padded_inputs, dtype=torch.float32)
         
            # Create a mask for non-padding values
            padding_mask = tensor_inputs != -1  # Exclude padding values
            range_mask = torch.zeros_like(tensor_inputs, dtype=torch.bool)
            range_mask[18:119] = True  #avoid normalizing class_distribution since already normalized

            # Combine both masks
            mask = padding_mask & ~range_mask
            if mask.any():
                tensor_inputs[mask] = (tensor_inputs[mask] - self.input_min[mask]) / (
                    self.input_max[mask] - self.input_min[mask] + 1e-8)
                
            return key, tensor_inputs, label_tensor_norm
    
    def filter_test_data(self, dataset_name: str = 'food101', discard_percentage: int = 50):
        test_data = {}

        for key, value in self.valid_data.items():
            # Unpack the key to extract model_name, dataset_name, and discard_percentage
            model_name, data_name, discard, batch_size, learning_rate = key

            # Check conditions for filtering
            if data_name == dataset_name and discard == discard_percentage:
                test_data[key] = value

        return test_data

    def compute_statistics(self):
        all_inputs = []
        target_keys = ['val_acc', 'energy_cumulative'] 


        for key, value in self.valid_data.items():
        # Extract input keys excluding the target keys
            input_keys = [k for k in value.keys() if k not in target_keys]
            inputs = {k: value[k] for k in input_keys}

            # Pad the sample for inputs and collect it
            padded_sample = self.pad_single_sample(inputs)
            tensor_value = torch.tensor(padded_sample, dtype=torch.float32)
            all_inputs.append(tensor_value)
            # Stack into a 2D tensor (samples x features)
        all_inputs = torch.stack(all_inputs)  # Shape: (num_samples, num_features)

        # Create a mask to exclude padding values (-1)
        mask = all_inputs != -1  # Shape: (num_samples, num_features)

        self.input_min = torch.min(all_inputs.masked_fill(~mask, float('inf')), dim=0).values
        self.input_max = torch.max(all_inputs.masked_fill(~mask, float('-inf')), dim=0).values

        label_masks = []  # Collect label masks
        all_labels=[]

        # Iterate through valid_data and pad all labels to 100
        for key, value in self.valid_data.items():
            labels = []
            masks = []  # To store masks for each label field
            for k in target_keys:
                if k in value:
                    # Handle the special case where sequence length is 101
                    max_len = 400
                    sequence = value[k]

                    if len(sequence)>400:
                        sequence = sequence[:400]

                    if len(sequence) == 101:
                        # Truncate the sequence to 100
                        sequence = sequence[:100]
                    
                    # Pad the sequence to 400
                    padded = np.pad(
                        sequence,
                        (0, max(0, max_len - len(sequence))),
                        constant_values=-1
                    )
                    tensor_label = torch.tensor(padded, dtype=torch.float32)
                    labels.append(tensor_label)
                    masks.append(tensor_label != -1)  # Create a mask for non-padding values
            if labels:
                stacked_labels = torch.stack(labels, dim=-1)  # Shape: (seq_len, num_targets)
                stacked_masks = torch.stack(masks, dim=-1)  # Corresponding mask
                all_labels.append(stacked_labels)  # Append the tensor to the list
                label_masks.append(stacked_masks)

        # Concatenate all labels and masks if the list is not empty
        if all_labels:
            all_labels = torch.cat(all_labels, dim=0)  # Combine all samples
            label_masks = torch.cat(label_masks, dim=0)  # Combine all masks
            # Compute min and max values for non-padding elements
            self.label_min = torch.min(all_labels.masked_fill(~label_masks, float('inf')), dim=0).values
            self.label_max = torch.max(all_labels.masked_fill(~label_masks, float('-inf')), dim=0).values
        else:
            self.label_min = torch.tensor([])  # Empty tensor if no valid data
            self.label_max = torch.tensor([])

        # Output the results
        print(f"Label Min: {self.label_min}")
        print(f"Label Max: {self.label_max}")


    def filter_valid_experiments(self):
        target_keys = ['val_acc','energy'] #'emissions_codecarbon']#, 'emissions_eco2ai']
        for key, value in self.entire_data.items():
            valid = True
            label_shapes = []

            # Determine the expected target length based on the model name
            model_name = key[0]  
            if model_name in ["bert-base-uncased", "roberta-base","microsoft_phi-2", "mistralai_Mistral-7B-v0.3"]:
                expected_length = 5
                check_condition = lambda x: len(x) == expected_length
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

        # Print summary of exclusions
        print("Excluded experiments due to invalid label shapes or values:")
        for exp_name, details in self.invalid_experiments.items():
            print(f"  Experiment: {exp_name}, Label shapes: {details['label_shapes']}, Invalid fields: {details.get('invalid_fields', [])}")
        print(f'len valid data: {len(self.valid_data)}')
        print(f'len invalid data: {len(self.invalid_experiments)}')
        print(f'len total exps: {len(self.entire_data)}')



    def process_dict(self, single_feature):
        if isinstance(single_feature, dict):
            return {k: self.process_dict(v) for k, v in single_feature.items()}
        elif isinstance(single_feature, list):
            return [self.process_dict(v) for v in single_feature]
        # elif isinstance(single_feature, (int, float)):
        #     return torch.tensor(single_feature)
        else:
            return single_feature
        
    
    def pad_single_sample(self, data_item):
        """
        Pad a single data item based on max_padding.
        """
        combined_padded_data = []

        for field, max_len in self.max_padding.items():
            if field in data_item:
                # Handle list-type fields
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
                        print(f"Invalid raw data in field '{field}': {flattened_field_data}")

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
                        print(f"Invalid tensor data in field '{field}': {tensor_data}")

                    padded_field = np.pad(
                        tensor_data[:max_len],
                        (0, max(0, max_len - len(tensor_data))),
                        constant_values=-1
                    )
                    combined_padded_data.extend(padded_field.tolist())
                else:
                    combined_padded_data.append(float(data_item[field]))

        # Append one-hot encoding for activation functions
        one_hot_activation = [0, 0, 0, 0]  # [ReLU, Sigmoid]
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
        Compute the maximum length for each list key across the dataset, excluding 'activation_functions'.
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
        
        NOT_FOUND = False
        results_folder = Path(load_yaml_exp_folder()[2])                                                                                   
        folder_name = os.path.join(results_folder, model_name, f'{dataset_name.lower()}_{exp_name}') #path da modificare
        folder_name = results_folder / model_name / f"{dataset_name.lower()}_{exp_name}"
        #print(f"Checking folder: {folder_name}")

        try:
            if model_name in ["bert-base-uncased", "roberta-base", "microsoft_phi-2", "mistralai_Mistral-7B-v0.3"]:
                # For bert-base-uncased and roberta-base, read from training_log.csv
                metrics_df = pd.read_csv(f'{folder_name}/epochs_results/training_log.csv')
                val_accuracy_df = metrics_df[metrics_df['Metric_Name'] == 'val_accuracy']
                # Extract 'Value' column as a list
                metrics = val_accuracy_df['Value'].dropna().values.tolist()
                #print(metrics)
            else:
                # Standard path for val_acc
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
                print(f'No metrics found for {model_name} on {dataset_name} with {exp_name}')

        if not NOT_FOUND:
            emissions_codecarbon = pd.read_csv(f'{folder_name}/emissions.csv')['emissions'].values.tolist()
            energy=pd.read_csv(f'{folder_name}/emissions.csv')['energy_consumed'].values.tolist()
            emissions_eco2ai = pd.read_csv(f'{folder_name}/emission_eco2ai.csv')['CO2_emissions(kg)'].values.tolist()
            return torch.tensor(metrics), torch.tensor(emissions_codecarbon), torch.tensor(energy),torch.tensor(emissions_eco2ai)
        else:
            return torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
        
    def pad_sequence(self):
        """
        Pad all dataset items and return them as a list of numpy arrays.
        """
        self.max_padding = self.compute_max_padding()
        return [self.pad_single_sample(data_item) for data_item in self.valid_data.values()]

    def validate_padded_output(self):
        # Display the first 20 keys in valid_data
        print("\n=== First Keys in valid_data ===")
        for i, key in enumerate(self.valid_data.keys()):
            if i >=2 :  
                break
            print(f"Validating key: {key}")

            # Get all padded outputs and the specific padded output for this key
            padded_outputs = self.pad_sequence()  # Get all padded outputs
            padded_output = padded_outputs[i]  # Corresponding padded output for the current key
            data_item = self.valid_data[key]  # Original data item

            # Validate if all padded outputs have the same shape
            first_shape = padded_outputs[0].shape
            consistent_shape = all(output.shape == first_shape for output in padded_outputs)

            if not consistent_shape:
                print(f"Mismatch in shape detected for key {key}. Expected {first_shape}.")
            else:
                print("All padded outputs have the same shape.")

            # Validate individual fields
            print("\n=== Validation of Individual Fields ===")
            index = 0

            for field, max_len in self.max_padding.items():
                if field in data_item:
                    raw_value = data_item[field]
                    if isinstance(raw_value, list):
                        padded_value = padded_output[index:index + max_len]
                        index += max_len
                        print(f"{field}:")
                        print(f"  Raw: {raw_value[:105]} (len={len(raw_value)})")  
                        print(f"  Padded: {padded_value[:105]} (len={len(padded_value)})") 
                    elif isinstance(raw_value, torch.Tensor):
                        raw_value = raw_value.cpu().numpy().tolist()
                        padded_value = padded_output[index:index + max_len]
                        index += max_len
                        print(f"{field}:")
                        print(f"  Raw: {raw_value[:105]} (len={len(raw_value)})")
                        print(f"  Padded: {padded_value[:105]} (len={len(padded_value)})")
                    else:
                        padded_value = padded_output[index]
                        index += 1
                        print(f"{field}:")
                        print(f"  Raw: {raw_value}")
                        print(f"  Padded: {padded_value}")
                else:
                    padded_value = padded_output[index:index + max_len]
                    index += max_len
                    print(f"{field}:")
                    print(f"  Not present in data. Padded: {padded_value[:30]} (len={len(padded_value)})")

    def obtain_entire_dataset(self):
        final_data = {}
        all_models = get_models()
        results_folder = Path(load_yaml_exp_folder()[2])
 

        for model_folder in results_folder.iterdir():
            if not model_folder.is_dir() or model_folder.name not in all_models:
                continue

            model_name = model_folder.name
            for experiment_folder in model_folder.iterdir():
                if not experiment_folder.is_dir():
                    continue

                try:
                    # Split the folder name into parts
                    parts = experiment_folder.name.split('_')

                    # Determine how to extract dataset_name and other parameters
                    if model_name in ["bert-base-uncased", "roberta-base","microsoft_phi-2", "mistralai_Mistral-7B-v0.3"]:
                        parts = experiment_folder.name.split('_')

                        # Ensure the folder has the expected structure
                        if len(parts) < 5:
                            raise ValueError(f"Folder name does not have enough parts: {experiment_folder.name}")
                        
                        dataset_name = f"{parts[0]}_{parts[1]}"
                        
                        
                        # Validate each part before conversion
                        if parts[2] != "discard":
                            raise ValueError(f"Unexpected value in parts[2]: {parts[2]} in folder {experiment_folder.name}")
                        
                        discard_percentage = int(parts[3]) # Convert parts[3] to int (this is safe now)
                        discard_percentage_forcomp= discard_percentage 
                        batch_size = 1  # Default batch size for these models
                        learning_rate = float(parts[4])  # Convert parts[4] to float
                        exp_name=f"discard_{discard_percentage}_{learning_rate}"
                        train_perc=0.8
                    else:
                        parts = experiment_folder.name.split('_')
                        if len(parts) < 5:
                            raise ValueError(f"Folder name does not have enough parts: {experiment_folder.name}")

                        dataset_name = parts[0]
                        discard_percentage = int(parts[2])
                        train_perc=0.7
                        if model_name in ["BERT4Rec", "CORE", "GRU4Rec", "SASRec"]:
                            discard_percentage_forcomp=discard_percentage
                        else: 
                            discard_percentage_forcomp= 100-discard_percentage 
 
                        batch_size = int(parts[3])
                        learning_rate = float(parts[4])
                        exp_name=f"discard_{discard_percentage}_{batch_size}_{learning_rate}"
        

                    keys = (model_name, dataset_name, discard_percentage, batch_size, learning_rate)

                    if dataset_name not in self.data_names:
                        raise KeyError(f"Dataset name {dataset_name} not found in self.data_names.")


                    final_data[keys] = {}
                    final_data[keys]['FLOPS'] = extract_flops_from_text(experiment_folder) if experiment_folder.exists() else -1
                    final_data[keys]['depth'], final_data[keys]['params'] = extract_architecture_metrics(experiment_folder) if experiment_folder.exists() else -1  #latency ignored because we have few data

                    final_data[keys]['discard_percentage'] = discard_percentage_forcomp
                    final_data[keys]['batch_size'] = batch_size
                    final_data[keys]['learning_rate'] = learning_rate



                    res = self.obtain_precomputed_results(
                        model_name=model_name, dataset_name=dataset_name,
                        exp_name=exp_name
                    )

                    final_data[keys]['val_acc'], final_data[keys]['emissions_codecarbon'],final_data[keys]['energy'], final_data[keys]['emissions_eco2ai'] = res
                    final_data[keys]['energy_cumulative'] = torch.cumsum(final_data[keys]['energy'], dim=0)

                    final_data[keys].update(self.data_dict[self.data_names[dataset_name]]) #dataset info dict
                    final_data[keys]['num_val_examples'] = -1

                    # Check if `num_train_examples` is empty
                    if not final_data[keys]['num_train_examples']:
                        final_data[keys]['num_train_examples']=-1
                    else:
                        # Make copies of the original data to avoid overwriting
                        num_train_examples_copy = final_data[keys]['num_train_examples']

                        # Perform calculations on the copies
                        final_data[keys]['num_val_examples'] = (
                            num_train_examples_copy * (100 - discard_percentage_forcomp) / 100 * (1 - train_perc)
                        )
                        final_data[keys]['num_train_examples'] = (
                            num_train_examples_copy * (100 - discard_percentage_forcomp) / 100 * train_perc
                        )
                    if model_name in ["BERT4Rec", "CORE", "GRU4Rec", "SASRec"]:
                        num_users_copy= final_data[keys]['num_users']
                        final_data[keys]['num_users']=num_users_copy*(100-discard_percentage_forcomp)/100

                    layers_info = parse_layers_info(experiment_folder) if experiment_folder.exists() else -1
                    for k, v in layers_info.items():
                        final_data[keys][k] = v

                    if 'pooling_layers' in final_data[keys]:
                        del final_data[keys]['pooling_layers']
                    del final_data[keys]['emissions_eco2ai']
                    del final_data[keys]['emissions_codecarbon']
                    del final_data[keys]['energy']
                    del final_data[keys]['num_test_examples']
                    for keys in final_data.keys():
                        final_data[keys].pop('task', None)
                
                except Exception as e:
                    print(f"Error processing folder {experiment_folder}: {e}. Skipped experiment")
                    if keys in final_data:  # Ensure incomplete entries are removed
                         del final_data[keys]
                    continue

                

        return final_data


if __name__ == '__main__':
    with open(load_yaml_exp_folder()[0], 'rb') as handle:
        datasets_to_features = pickle.load(handle)
    with open(load_yaml_exp_folder()[1], 'rb') as handle:
        models_to_features = pickle.load(handle)

       # Initialize full dataset
    full_dataset = ArchitectureDataset(models_to_features, datasets_to_features)
    #full_dataset.get_observation_with_min_label()
    full_dataset.validate_padded_output()

    # Define proportions for training and validation
    val_split_ratio = 0.2  # 20% for validation
    train_size = int(len(full_dataset) * (1 - val_split_ratio))
    val_size = len(full_dataset) - train_size

    # Split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    

    # Initialize data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Verify train and validation datasets
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Example of iterating over the training dataloader
    for batch_idx, (key, inputs, labels) in enumerate(train_dataloader):
        if batch_idx == 0:

            print(f"Batch {batch_idx}:")
            print(f"key: {key}")
            # Check and print the shape of the inputs
            print(f"  Inputs shape: {inputs.shape}")
            torch.set_printoptions(threshold=torch.inf)
            print(f'Input:{inputs} ')
            # Check and print the shapes of the labels
            print (f'labels shape {labels.shape}')
            #for label_idx, label in enumerate(labels):
              #  print(f"  Label {label_idx} shape: {label.shape}, label {label}, unique: {len(torch.unique(label))}")
