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


def get_models(num_models: int = -1, models_to_ignore: list = []):
    available_models = [
        "resnet18",
        ### "resnet101",
        "alexnet",
        "vgg16",
        "squeezenet",
        "efficientnet",
        "vit",
        ### "mobilenet",
        ### "swin_transformer",
    ]
    if num_models == -1:
        return available_models
    else:
        if len(models_to_ignore) > 0:
            available_models = [
                model for model in available_models if model not in models_to_ignore
            ]
            return available_models
        else:
            return available_models[:num_models]

def get_all_datasets() -> list:
    return ["food101", "cifar10", "cifar100", "mnist"] 


def generate_lr(start_order, end_order, num_samples, seed:int = 42):
    if seed is not None:
        np.random.seed(seed)  
    values = np.logspace(start_order, end_order, num_samples)
    return values + np.random.rand(num_samples) * 1e-4  