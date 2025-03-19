"""
Set of utility functions used for model training, dataset management, and parameter generation.
"""

import torch
import pytorch_lightning as pl
import numpy as np
import os
import json

def get_models(models_type:str='all'):
    available_models = [
        #"resnet18",
        #"alexnet",
        #"vgg16",
        #"squeezenet",
        #"efficientnet",
        #"vit",
        #"BERT4Rec",
        #"CORE", 
       # "GRU4Rec",
        #"SASRec",
        #"roberta-base",
        #"bert-base-uncased",
        #'microsoft_phi-2',
        #'mistralai_Mistral-7B-v0.3'
    ]
    if models_type == "all":
        return available_models
    elif models_type == "vision":
        return available_models[:6]
    elif models_type == "recommendation":
        return available_models[6:9]
    elif models_type == "text":
        return available_models[10:]
    else:
        raise ValueError("Invalid model type")

def get_all_datasets(dataset_type:str="vision") -> list:
    if dataset_type == "vision":
        return ["food101", "cifar10", "fashionmnist", "mnist"]
    elif dataset_type == "text":
        return ["cornell-movie-review-data/rotten_tomatoes", "google/boolq", "dair-ai/emotion", "stanfordnlp/imdb"]
    else:
        raise ValueError("Invalid dataset type")

def load_json_results(folder_path, starter='results_MAE'):
    results = []
    for filename in os.listdir(folder_path):
        if filename.startswith(starter) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as json_file:
                results.append(json.load(json_file))
    return results

def seed_everything(seed: int):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    This function seeds the following libraries: `random`, `os`, `numpy`, `torch`, 
    and PyTorch Lightning, ensuring deterministic behavior across different environments.

    Args:
        seed (int): The seed value to be used for all random number generators.
    """
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


def compute_model_params(model):
    """
    Computes the total number of trainable parameters in a given model.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters will be counted.

    Returns:
        int: The total number of parameters in the model.
    """
    model_parameters = sum(p.numel() for p in model.parameters())
    return model_parameters


def generate_lr(start_order, end_order, num_samples, seed:int = 42):
    """
    Generates a list of learning rates sampled logarithmically within a given range.

    Args:
        start_order (float): The exponent for the starting value in the range (e.g., `1e-3` is represented by -3).
        end_order (float): The exponent for the ending value in the range (e.g., `1e-1` is represented by -1).
        num_samples (int): The number of learning rate values to generate.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        np.ndarray: An array of learning rate values sampled within the specified range.
    """
    if seed is not None:
        np.random.seed(seed)  
    values = np.logspace(start_order, end_order, num_samples)
    return values + np.random.rand(num_samples) * 1e-4  

def denormalize_outputs(normalized_output, label_min, label_max):
    """
    Converts normalized outputs back to their original scale using the provided min and max values.

    Args:
        normalized_output (np.ndarray): The normalized outputs to be denormalized.
        label_min (float): The minimum value of the original scale.
        label_max (float): The maximum value of the original scale.

    Returns:
        np.ndarray: The denormalized outputs.
    """
    return normalized_output * (label_max - label_min) + label_min

    
