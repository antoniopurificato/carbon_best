import torch
import pytorch_lightning as pl
import numpy as np

def seed_everything(seed: int):
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
    model_parameters = sum(p.numel() for p in model.parameters())
    return model_parameters


def get_models(models_type:str='all'):
    available_models = [
        "resnet18",
        "alexnet",
        "vgg16",
        "squeezenet",
        "efficientnet",
        "vit",
        "BERT4Rec",
        "CORE", 
        "GRU4Rec",
        "SASRec",
        "roberta-base",
        "bert-base-uncased",
        'microsoft/phi-2',
        'mistralai/Mistral-7B-v0.3'
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
        return ["food101", "cifar10", "cifar100", "mnist"]
    elif dataset_type == "text":
        return ["cornell-movie-review-data/rotten_tomatoes", "google/boolq", "dair-ai/emotion", "stanfordnlp/imdb"]


def generate_lr(start_order, end_order, num_samples, seed:int = 42):
    if seed is not None:
        np.random.seed(seed)  
    values = np.logspace(start_order, end_order, num_samples)
    return values + np.random.rand(num_samples) * 1e-4  

def denormalize_outputs(normalized_output, label_min, label_max):
    return normalized_output * (label_max - label_min) + label_min