import os
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

from src.utils import *

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

def extract_flops_from_text(file_path:str): 
    file_path = os.path.join(file_path, 'train_flops.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    try:
        with open(file_path, 'r') as file:
            content = file.read()
        pattern = r'fwd flops of model = fwd flops per GPU \* mp_size: (.*?)T'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            extracted_value = float(match.group(1).strip())
            return extracted_value
        else:
            return None
    except Exception as e:
       raise FileNotFoundError(f"Error reading the file '{file_path}': {e}")

def load_yaml(network_name, dataset_name):
    file_path = os.path.join(base_path, network_name, dataset_name, 'results.yml')

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return None

    try:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        return yaml_content
    except Exception as e:
        print(f"Error reading the file '{file_path}': {e}")
        return None


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


