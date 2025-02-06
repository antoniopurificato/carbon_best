"""
Utility Functions for Experiment Data Extraction and Analysis
=============================================================

This module provides a suite of functions and utilities to support the reading,
parsing, and analysis of experimental results stored in a particular folder 
structure. These functions include:

1. **YAML-based Folder Loading:** 
   - Dynamically loads folder paths from a YAML configuration file.

2. **Experiment Status Checking:**
   - Identifies which experiments have been run for which models and collates 
     them for analysis.

3. **Hardware and Architecture Parsing:**
   - Extracts hardware-related information (e.g., GPU, CPU details) from text files.
   - Retrieves architecture-level metrics (FLOPs, depth, parameter counts).
   - Parses neural network layers (e.g., convolutional, attention, dropout layers).

File Input/Output:
------------------
- **Input:** YAML files specifying folder paths, text files storing experiment 
  architectures (e.g., train_flops.txt), CSV files for logging metrics, etc.
- **Output:** In-memory data structures (dictionaries, lists) capturing model 
  experiment IDs, hardware info, or architecture-specific layer details.
"""

import os
import re
import pandas as pd
import os
import yaml
import ast
from pathlib import Path
import warnings
import subprocess
import gdown

from src.utils.secondary_utils import *

def download_data(url='1rptEKvfebpp-C2BRkIR_dy8whgOuyBKO',
                 output='src/KB_TEST.zip'):
    gdown.download(id=url, output=output)
    subprocess.run(['unzip', '-o', output, '-d', 'src/'], check=True)

def load_yaml_exp_folder(yaml_name:str='configs/folders.yaml',
                        data_pkl_name:str='datasets_to_features_dict_TOT_std.pkl',
                        model_pkl_name:str='models_to_features_dict_new.pkl',
                        base_path:str='src'):
    
    """
    Loads folder paths and file references from a YAML configuration file.

    Args:
        yaml_name (str, optional):
            Name of the YAML file containing folder paths. Defaults to 'configs/folders.yaml'.
        data_pkl_name (str, optional):
            Name of the data pickle file. Defaults to 'datasets_to_features_dict_TOT_std.pkl'.
        model_pkl_name (str, optional):
            Name of the model pickle file. Defaults to 'models_to_features_dict_new.pkl'.
        base_path (str, optional):
            Base directory path where the YAML file is located. Defaults to 'src/utils'.

    Returns:
        tuple: A tuple containing:
            - (str) Path to the data pkl file.
            - (str) Path to the model pkl file.
            - (str) Path to the folder where experiment results are stored.
    """
    with open(os.path.join(base_path, yaml_name), 'r') as file:
        folders = yaml.safe_load(file)
    return os.path.join(folders['pkl_folder'], data_pkl_name), os.path.join(folders['pkl_folder'], model_pkl_name), folders['results_folder']
        

def check_experiments_status(results_path:str='results'):
    """
    Summarizes the experiment status for each model folder by counting and listing 
    experiment subfolders.

    Args:
        results_path (str, optional):
            The path to the directory containing model experiment folders. Defaults to 'results'.

    Returns:
        str: A formatted string containing:
            - The total number of experiments found for each model.
            - Additional details (experiment IDs) for debugging or auditing.
    """
    final = ""
    additional_infos = ""
    model_dirs = [os.path.join(results_path, model) for model in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, model))]    
    for model_dir in model_dirs:
        additional_infos += f"**{model_dir.replace('results/', '')}**\n"
        experiment_ids = get_experiment_ids(model_dir)
        experiment_ids = [exp for exp in experiment_ids if "version_0" not in exp or "checkpoint-" not in exp]
        experiment_ids = [s for s in experiment_ids if s.count('_') >= 2]
        for exp in experiment_ids:
            additional_infos += f"- {exp}\n"
        final += f"Model: {model_dir.replace('results/', '')} - Experiments done: {len(experiment_ids)}\n"
    return final + '\n--------**Additional infos**-----------\n' +additional_infos

def get_experiment_ids(path):
    """
    Gathers all unique experiment IDs by walking through a directory tree.

    The experiment ID is derived from the basename of each subfolder containing files.

    Args:
        path (str): The root path to search for experiment subfolders.

    Returns:
        set: A set of unique experiment IDs, inferred from folder structures.
    """
    experiment_ids = set()
    for root, _, files in os.walk(path):
        for file in files:
            experiment_ids.add(os.path.basename(root))
    return experiment_ids

def get_exps_in_common(results_path:str='results'):
    """
    Identifies experiments that are common across all recognized models 
    (models returned by get_models() in the secondary utils).

    Args:
        results_path (str, optional):
            The base directory containing all model subfolders. Defaults to 'results'.

    Returns:
        list: A list of experiment IDs that are present in every recognized model's folder.
    """
    model_dirs = [os.path.join(results_path, model) for model in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, model)) and model in get_models()]    
    common_experiment_ids = None
    for model_dir in model_dirs:
        experiment_ids = get_experiment_ids(model_dir)
        if common_experiment_ids is None:
            common_experiment_ids = experiment_ids
        else:
            common_experiment_ids &= experiment_ids

    experiments_list = [exp for exp in common_experiment_ids if exp not in {'version_0', 'checkpoints'}]
    return experiments_list

    
def get_folder_name(load_yaml_exp_folder, model_name, dataset_name, exp_name):
    """
    Constructs the full path to an experiment folder based on YAML-specified 
    result directories.

    Args:
        load_yaml_exp_folder (callable):
            A function (usually this module's load_yaml_exp_folder) that returns 
            the necessary paths from a YAML config file.
        model_name (str):
            The model subfolder name.
        dataset_name (str):
            The dataset subfolder name (in lowercase).
        exp_name (str):
            The experiment folder name appended to the dataset name.

    Returns:
        str: The absolute path to the experiment's folder.
    """
    results_folder = Path(load_yaml_exp_folder()[2])
    folder_name = results_folder / model_name / f"{dataset_name.lower()}_{exp_name}"
    return str(folder_name)

def extract_hardware_info_from_file(data, keys):
    """
    Parses a text file (in dictionary format) to extract specified hardware fields.

    Args:
        file_path (str):
            Path to the hardware info file (expected to contain a dictionary-like string).
        keys (list):
            The list of fields to extract from the parsed dictionary.

    Returns:
        dict: A dictionary of {key: value} for all fields specified in `keys`.
              If a field is missing, the result will contain `None` or the raw 
              value found in the file. Also attempts to convert string fields like 
              'XX MB/GB' to an integer.
    """
    #with open(file_path, "r") as file:
    #data = ast.literal_eval(file.read())  # Parse the file as a dictionary
    result = {}
    for key in keys:
        value = data.get(key)
        # Convert "total_memory" and "L2_cache_size" to numeric values
        if isinstance(value, str) and "MB" in value:
            result[key] = int(value.split("MB")[0].strip())
        elif isinstance(value, str) and "GB" in value:
            result[key] = int(value.split("GB")[0].strip())
        else:
            result[key] = value
    return result

def extract_flops_from_text(file_path: str):
    """
    Extracts FLOPs (floating-point operations) information from a text file named 'train_flops.txt' 
    located in the given directory path.

    Args:
        file_path (str):
            The directory path containing 'train_flops.txt'.

    Returns:
        float: The extracted FLOPs value. Returns -1 if the information is not found or if
               there's an error converting the found value to float.

    Raises:
        FileNotFoundError: If 'train_flops.txt' does not exist in the given directory.
        RuntimeError: If an error occurs while reading the file.
    """
    file_path = os.path.join(file_path, 'train_flops.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    try:
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Adjusted pattern to be more flexible
        pattern = r'fwd flops of model = fwd flops per GPU \* mp_size:\s*([\d\.]+)[Tt]?'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            try:
                extracted_value = float(match.group(1).strip())
                return extracted_value
            except ValueError:
                print(f"Error converting extracted value to float in file '{file_path}'.")
                return -1
        else:
            print(f"FLOPS information not found in file '{file_path}'. Returning -1.")
            return -1

    except Exception as e:
        raise RuntimeError(f"Error reading the file '{file_path}': {e}")


def extract_architecture_metrics(file_path: str):
    """
    Extracts depth and parameter counts from 'train_flops.txt' within the given directory.

    The function looks for lines indicating the model's depth (lines containing "depth xx")
    and a parameter count pattern "params per GPU: xxx (K/M/B)".

    Args:
        file_path (str):
            The directory path containing 'train_flops.txt'.

    Returns:
        tuple: (depth, params)
            - depth (int): The total count of recognized depth lines.
            - params (float): The number of parameters in millions. 
                              If a 'B' suffix is found, it is converted to thousands of millions.
                              If a 'K' suffix is found, it is converted to fractions of a million (K=0.001M).
                              Returns -1 if the parameters are not found.

    Raises:
        FileNotFoundError: If 'train_flops.txt' is not present.
    """
    file_path = os.path.join(file_path, 'train_flops.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Extract Total Depth
        depth_matches = re.findall(r"depth (\d+):", content)
        depth= len(depth_matches)

        params_pattern = r'params per GPU:\s*([\d\.]+)\s*([KMB])'
        params_match = re.search(params_pattern, content, re.DOTALL)

        if params_match:
            params_value = float(params_match.group(1).strip())
            unit = params_match.group(2).strip()

            # Convert K to M if necessary
            if unit == "K":
                params = params_value / 1000  # Convert thousands to millions
            elif unit == "M":
                params = params_value
            elif unit == "B":
                params = params_value * 1000

        else:
            params = -1
            print("Parameters per GPU not found in the file. Placeholder -1 inserted")

        return depth, params 

    except Exception as e:
       raise FileNotFoundError(f"Error reading the file '{file_path}': {e}")


def extract_attention_layers(content):
    """
    Searches for various self-attention modules in a string (content of 'train_flops.txt') and 
    extracts their in_features, inferred num_heads, and LoRA (low-rank adaptation) configuration 
    if present.

    Args:
        content (str):
            The raw text content from a 'train_flops.txt' file.

    Returns:
        list: A list of dictionaries, each describing an attention layer with keys such as:
              {
                  "type": <str>,
                  "in_features": <int>,
                  "num_heads": <int>,
                  "lora_r": <int or -1 if not present>
              }
    """
    # Define patterns for various attention types
    attention_layer_patterns = [
        (
            "MultiheadAttention",
            re.compile(
                r"MultiheadAttention\([\s\S]*?in_features=(\d+)[\s\S]*?out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
        ),
        (
            "BertSelfAttention",
            re.compile(
                r"(?:BertSelfAttention)[\s\S]*?\(query\): Linear\([\s\S]*?in_features=(\d+), out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
        ),
        (
            "RobertaSelfAttention",
            re.compile(
                r"(?:RobertaSelfAttention)[\s\S]*?\(query\): Linear\([\s\S]*?in_features=(\d+), out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
        ),
        (
            "PhiSdpaAttention",
            re.compile(
                 r"PhiSdpaAttention[\s\S]*?\(q_proj\)[\s\S]*?in_features=(\d+).*?out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
        ), 
        (   "MistralSdpaAttention",
              re.compile(
                 r"MistralSdpaAttention[\s\S]*?\(q_proj\)[\s\S]*?in_features=(\d+).*?out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
         )
    ]

    # Pattern specifically for lora_A
    lora_a_pattern = re.compile(
        #r"PhiSdpaAttention|MistralSdpaAttention[\s\S]*?\(lora_A\)[\s\S]*?out_features=(\d+)",
        r"lora_A[\s\S]*?out_features=(\d+)",
        re.MULTILINE | re.IGNORECASE)

    attention_layers = []

    for attention_type, pattern in attention_layer_patterns:
        matches = pattern.findall(content)
        if matches:
            for match in matches:
                # Extract in_features and out_features
                in_features, out_features = int(match[0]), int(match[1])

                # Derive num_heads (assuming head_dim = 64)
                head_dim = 64
                num_heads = in_features // head_dim
                if num_heads<=0:
                    num_heads=1


                # Extract the r value from lora_A
                lora_a_match = lora_a_pattern.search(content)
                
                r_value = int(lora_a_match.group(1)) if lora_a_match else -1

                # Add to the flattened list
                attention_layers.append({
                    "type": attention_type,
                    "in_features": in_features,
                    "num_heads": num_heads,
                    "lora_r": r_value  # Only the r value (out_features of lora_A)
                })

            # Stop further processing once matches are found
            break

    return attention_layers




def extract_dropout_layers(content):
    """
    Extracts all Dropout layers from the provided text content and retrieves their dropout probabilities.

    Args:
        content (str):
            The raw text from 'train_flops.txt' detailing model architecture.

    Returns:
        list: A list of dictionaries, each containing:
              {"p": <float> } for the dropout probability.
    """
    dropout_pattern = re.compile(
        r"Dropout\(.*?p=([\d.]+)", re.MULTILINE | re.IGNORECASE
    )

    matches = dropout_pattern.findall(content)
    #print(f"Matches found for dropout : {matches}")
    dropout_layers = []

    for match in matches:
        dropout_layers.append({
            "p": float(match)
        })

    return dropout_layers



def extract_conv_layers(content):
    """
    Extracts convolutional layers from the content of 'train_flops.txt' by matching 
    lines of the form: Conv2d(in_ch, out_ch, kernel_size=(kx, ky), stride=(sx, sy)).

    Args:
        content (str):
            The string content of 'train_flops.txt'.

    Returns:
        list: A list of dictionaries, each describing a convolutional layer with fields:
              {
                  "output_channels": <int>,
                  "kernel_size": (kx, ky),
                  "stride": (sx, sy)
              }
    """
    conv_proj_pattern = re.compile(
        r"Conv2d\([^)]*?,\s*(\d+),\s*(\d+),.*?kernel_size=\((\d+),\s*(\d+)\),\s*stride=\((\d+),\s*(\d+)\)",
        re.MULTILINE | re.IGNORECASE
    )

    matches = conv_proj_pattern.findall(content)
    #print(f"Matches found for conv layer : {matches}")
    embedding_parameters = []

    for match in matches:
        embedding_parameters.append({
            #"input_channels": int(match[0]),
            "output_channels": int(match[1]),
            "kernel_size": (int(match[2]), int(match[3])),
            "stride": (int(match[4]), int(match[5]))
        })

    return embedding_parameters

def extract_lora_layers(content):
    """
    Extracts LoRA (Low-Rank Adaptation) configurations such as lora_A, lora_B, and lora_dropout 
    from the given content string.

    Args:
        content (str):
            The raw text possibly containing references to LoRA modules.

    Returns:
        list: A list of dictionaries describing each LoRA layer, e.g.:
              {
                  "type": <"lora_A"/"lora_B"/"lora_dropout">,
                  "in_features": <int>,
                  "out_features": <int>,
                  "dropout": <float>
              }
    """
    
    # Define patterns to match LoRA configurations (lora_A, lora_B, lora_dropout)
    lora_pattern = re.compile(
        r"(lora_A|lora_B|lora_dropout):.*?Linear\(.*?in_features=(\d+),.*?out_features=(\d+).*?p=([\d.]+)?",
        re.MULTILINE | re.IGNORECASE
    )

    matches = lora_pattern.findall(content)
    lora_layers = []

    for match in matches:
        layer_type, in_features, out_features, dropout = match
        lora_layers.append({
            "type": layer_type,
            "in_features": int(in_features),
            "out_features": int(out_features),
            "dropout": float(dropout) if dropout else 0
        })

    return lora_layers


def parse_layers_info(file_path: str):
    """
    Parses multiple layer types (FC, attention, dropout, conv, embedding, etc.) 
    from a text file named 'train_flops.txt'.

    Args:
        file_path (str):
            The directory path containing 'train_flops.txt'.

    Returns:
        dict: A dictionary containing lists for each discovered layer category, e.g.:
              {
                  "conv_layers_NEW": [...],
                  "fc_layers_NEW": [...],
                  "attention_layers_NEW": [...],
                  "embedding_layers_NEW": [...],
                  "activation_functions_NEW": [...],
                  "batch_norm_layers_NEW": [...],
                  "layer_norm_layers_NEW": [...],
                  "dropout_NEW": [...]
              }

    Raises:
        FileNotFoundError: If 'train_flops.txt' is not found in the specified directory.
    """
    file_path = os.path.join(file_path, 'train_flops.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    grouped_data = {
        "conv_layers_NEW": [],
        "fc_layers_NEW": [],
        "attention_layers_NEW": [],
        "embedding_layers_NEW": [],
        "activation_functions_NEW": [],
        "batch_norm_layers_NEW": [],
        "layer_norm_layers_NEW":[]
        
    }

    # Updated patterns
    fc_layer_pattern = re.compile(r"Linear.*?in_features=(\d+), out_features=(\d+)")
    activation_pattern = re.compile(r"(ReLU|GELU|Sigmoid|Tanh|Softmax|Swish|SeLU)")
    batch_norm_pattern = re.compile(
        r"BatchNorm2d\(.*?, .*?, .*?, .*?, (\d+), eps=([\d.e-]+), momentum=([\d.e-]+)"
    )
    layer_norm_pattern = re.compile(
        r"LayerNorm.*?eps=([\d.e-]+)", re.MULTILINE | re.IGNORECASE
    )
    embedding_pattern = re.compile(
        r"Embedding\(.*?FLOPS.*?, (\d+), (\d+)", re.MULTILINE | re.IGNORECASE
    )

    with open(file_path, "r") as file:
        content = file.read()

    for fc_match in fc_layer_pattern.finditer(content):
        grouped_data["fc_layers_NEW"].append({
            "in_features": int(fc_match.group(1)) #,
           # "out_features": int(fc_match.group(2))
        })

    for activation_match in activation_pattern.finditer(content):
        grouped_data["activation_functions_NEW"].append({
            "type": activation_match.group(1)
        })

    for batch_norm_match in batch_norm_pattern.finditer(content):
        grouped_data["batch_norm_layers_NEW"].append({
            #"num_features": int(batch_norm_match.group(1)),
            "eps": float(batch_norm_match.group(2)),
            "momentum": float(batch_norm_match.group(3))
        })

    # Parse LayerNorm Layers
    for layer_norm_match in layer_norm_pattern.finditer(content):
        grouped_data["layer_norm_layers_NEW"].append({
            "eps": float(layer_norm_match.group(1))
        })

    # Parse Embedding Layers
    for embedding_match in embedding_pattern.finditer(content):
        grouped_data["embedding_layers_NEW"].append({
            "emb_dim": int(embedding_match.group(2))
        })
   
    grouped_data["attention_layers_NEW"]=extract_attention_layers(content)
    grouped_data["dropout_NEW"]=extract_dropout_layers(content)
    grouped_data["conv_layers_NEW"]=extract_conv_layers(content)

    return grouped_data


def get_pytorch_metrics(model_name='vit', exp_name='cifar100_discard_100_16_4.27022004702574e-05'):
    file_name = os.path.join('results', model_name, exp_name, 'version_0', 'metrics.csv')
    df = pd.read_csv(file_name)
    return df

def get_emission_metrics(model_name='vit', exp_name='cifar100_discard_100_16_4.27022004702574e-05', tracker='eco2ai'):
    if tracker == 'eco2ai':
        file_name = os.path.join('results', model_name, exp_name, 'emission_eco2ai.csv')
        df = pd.read_csv(file_name)['CO2_emissions(kg)']
        df.rename('Emissions eco2ai', inplace=True)
    if tracker == 'codecarbon':
        file_name = os.path.join('results', model_name, exp_name, 'emissions.csv')
        df = pd.read_csv(file_name)['emissions']
        df.rename('Emissions codecarbon', inplace=True)
    return df

def construct_experiment_name(meta: dict) -> str:
    """
    Builds the experiment folder name used in `obtain_precomputed_results` based on metadata.

    Args:
        meta (dict): Parsed metadata with fields like 'discard_pct', 'batch_size', 'lr', etc.

    Returns:
        str: The experiment name string.
    """
    if meta["is_llm"]:
        return f"discard_{meta['discard_pct']}_{meta['lr']}"
    return f"discard_{meta['discard_pct']}_{meta['batch_size']}_{meta['lr']}"

def process_energy_data(energy: torch.Tensor, model_name: str) -> torch.Tensor:
    """
    Trims or manipulates the energy data based on model_name. 
    Example: limiting the length for certain model types.

    Args:
        energy (torch.Tensor): The full energy measurement series.
        model_name (str): Name of the model to apply logic or slicing.

    Returns:
        torch.Tensor: Processed energy data (possibly truncated).
    """
    if model_name in {"bert-base-uncased", "roberta-base", "microsoft_phi-2", "mistralai_Mistral-7B-v0.3"}:
        return energy[:5]
    elif model_name in {"BERT4Rec", "CORE", "GRU4Rec", "SASRec"}:
        return energy[:400]
    else:
        return energy[:100]
    
def handle_processing_error(folder: Path, error: Exception, final_data: dict, key) -> None:
    """
    Called when an error occurs while processing an experiment folder. 
    Logs a warning and removes any partial/incomplete entry from final_data.

    Args:
        folder (Path): The experiment folder where the error occurred.
        error (Exception): The raised exception.
        final_data (dict): The dictionary of processed data so far.
        key: The unique key in final_data that might need cleanup.
    """
    warnings.warn(f"Error processing {folder}: {error}")
    if key in final_data:
        del final_data[key]

def is_valid_model_folder(folder: Path, all_models: list) -> bool:
    """
    Checks if a folder corresponds to a recognized model.

    Args:
        folder (Path): Path object referencing a potential model folder.
        all_models (list): List of valid model names.

    Returns:
        bool: True if folder is recognized and is a directory, otherwise False.
    """
    return folder.is_dir() and folder.name in all_models

def calculate_example_counts(final_data: dict, key: tuple, meta: dict) -> None:
    """
    Computes the adjusted number of training and validation examples 
    after discarding a percentage of data.

    Args:
        final_data (dict): The overall dictionary storing processed experiment data.
        key (tuple): Unique key for this experiment entry in final_data.
        meta (dict): Parsed metadata.
    """
    entry = final_data[key]
    if entry.get("num_train_examples"):
        total = entry["num_train_examples"]
        discard_factor = (100 - meta["discard_pct_comp"]) / 100
        entry["num_train_examples"] = total * discard_factor * meta["train_perc"]
        entry["num_val_examples"] = total * discard_factor * (1 - meta["train_perc"])
    else:
        entry["num_train_examples"] = -1
        entry["num_val_examples"] = -1

    # Adjust number of users for rec models
    if meta["is_rec_model"] and "num_users" in entry:
        entry["num_users"] = entry["num_users"] * (100 - meta["discard_pct_comp"]) / 100

def process_exp_folder(folder: Path) -> bool:
    """
    Verifies that the experiment folder is valid.

    Args:
        folder (Path): Path to an experiment folder.

    Returns:
        bool: True if it is a valid directory, otherwise False (with a warning).
    """
    if not folder.is_dir():
        warnings.warn(f"Skipping non-directory: {folder}")
        return False
    return True

def parse_experiment_metadata(exp_folder: Path, model_name: str) -> dict:
    """
    Parses the folder name to extract dataset, discard percentage, 
    learning rate, batch size, etc.

    Args:
        exp_folder (Path): Path to the experiment folder.
        model_name (str): Name of the model (e.g., 'bert-base-uncased').

    Returns:
        dict or None: A dictionary with parsed metadata if successful, else None.
    """
    try:
        # Adjust for amazon-beauty naming
        exp_name = exp_folder.name.replace("amazon_beauty", "amazon-beauty")
        parts = exp_name.split('_')

        meta = {
            "model_name": model_name,
            "is_llm": model_name in {
                "bert-base-uncased", 
                "roberta-base", 
                "microsoft_phi-2", 
                "mistralai_Mistral-7B-v0.3"
            },
            "is_rec_model": model_name in {"BERT4Rec", "CORE", "GRU4Rec", "SASRec"},
            "parts": parts
        }

        parse_dataset_info(meta)
        parse_training_params(meta)
        validate_metadata(meta)
        return meta

    except ValueError as e:
        warnings.warn(f"Invalid folder format {exp_folder.name}: {e}")
        return None

def parse_dataset_info(meta: dict) -> None:
    """
    Determines the dataset name from the split folder name parts.

    Raises:
        ValueError: If the folder structure doesn't match the expected format for LLMs.
    """
    parts = meta["parts"]
    if meta["is_llm"]:
        # Example: "dataset_subname_discard_30_0.0001"
        if len(parts) < 5:
            raise ValueError("Not enough parts for LLM folder naming.")
        meta["dataset"] = f"{parts[0]}_{parts[1]}"
        if parts[2] != "discard":
            raise ValueError("Expected 'discard' token in LLM folder name.")
    else:
        # Example: "dataset_discard_30_32_0.001"
        if len(parts) < 5:
            raise ValueError("Not enough parts for standard folder naming.")
        meta["dataset"] = parts[0]

def parse_training_params(meta: dict) -> None:
    """
    Extracts discard percentage, batch size, learning rate, and training percentage 
    from the folder name parts, depending on model type (LLM vs. rec model vs. others).

    Args:
        meta (dict): Metadata dictionary updated in-place.
    """
    parts = meta["parts"]

    if meta["is_llm"]:
        meta.update({
            "discard_pct": int(parts[3]),
            "batch_size": 1,  # LLM typically has batch size = 1
            "lr": float(parts[4]),
            "train_perc": 0.8
        })
    else:
        meta.update({
            "discard_pct": int(parts[2]),
            "batch_size": int(parts[3]),
            "lr": float(parts[4]),
            "train_perc": 0.7
        })

    # Comparative discard percentage: 
    # - For LLM or rec models, use same as discard_pct
    # - Otherwise, 100 - discard_pct
    if meta["is_llm"] or meta["is_rec_model"]:
        meta["discard_pct_comp"] = meta["discard_pct"]
    else:
        meta["discard_pct_comp"] = 100 - meta["discard_pct"]

def validate_metadata(meta: dict) -> None:
    """
    Basic sanity checks on metadata structure.
    Raises:
        ValueError: If metadata is inconsistent.
    """
    # Example checks
    if not meta["dataset"]:
        raise ValueError("Dataset name not parsed correctly.")
    if meta["discard_pct"] < 0 or meta["discard_pct"] > 100:
        raise ValueError("Discard percentage out of range (0-100).")