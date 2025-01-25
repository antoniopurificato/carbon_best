import os
import re
import pandas as pd
import os
import yaml

from src.utils.secondary_utils import *

def load_yaml_exp_folder(yaml_name:str='folders.yaml',
                        data_pkl_name:str='datasets_to_features_dict_TOT_std.pkl',
                        model_pkl_name:str='models_to_features_dict_new.pkl',
                        base_path:str='src/utils'):
    with open(os.path.join(base_path, yaml_name), 'r') as file:
        folders = yaml.safe_load(file)
    return os.path.join(folders['pkl_folder'], data_pkl_name), os.path.join(folders['pkl_folder'], model_pkl_name), folders['results_folder']
        

def check_experiments_status(results_path:str='results'):
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
    experiment_ids = set()
    for root, _, files in os.walk(path):
        for file in files:
            experiment_ids.add(os.path.basename(root))
    return experiment_ids

def get_exps_in_common(results_path:str='results'):
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

def extract_flops_from_text(file_path: str):
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
        r"PhiSdpaAttention|MistralSdpaAttention[\s\S]*?\(lora_A\)[\s\S]*?out_features=(\d+)",
        re.MULTILINE | re.IGNORECASE
    )

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

def get_time_per_epoch(model_name='vit', exp_name='cifar100_discard_100_16_4.27022004702574e-05'):
    file_name = os.path.join('results', model_name, exp_name, 'emissions.csv')
    df = pd.read_csv(file_name)['duration']
    return df

def concat_dfs(dfs):
    return pd.concat(dfs, axis=1)

def drop_undesired_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop)

if __name__ == "__main__":
    print(load_yaml_exp_folder())   