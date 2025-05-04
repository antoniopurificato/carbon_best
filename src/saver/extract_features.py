from collections import defaultdict
from typing import Optional
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from src.saver.extract_feature_utils import *
#import easy_rec
#from easy_rec import Dataset as EasyRecDataset


def extract_features(train_dataset, test_dataset, dataset_name: str, label_column: Optional[str] = None):
    features_list = [
        'num_train_examples', 'num_classes', 'class_distribution', 'mean_length',
        'mean_flesch_kincaid_grade', 'mean_dale_chall_readability_score', 'max_length',
        'task', 'density', 'num_users', 'mean', 'std', 'num_test_examples',
        'num_interactions', 'image_shape', 'num_items', 'avg_length', 'median_length'
    ]

    dict_of_features = {feature: defaultdict(lambda: None) for feature in features_list}

    if isinstance(train_dataset, TorchDataset) and isinstance(test_dataset, TorchDataset):
        dict_of_features = extract_features_from_torch_dataset(
            train_dataset, test_dataset, dataset_name, dict_of_features
        )

    elif isinstance(train_dataset, HFDataset) and isinstance(test_dataset, HFDataset):
        dict_of_features = extract_features_from_huggingface_dataset(
            train_dataset, test_dataset, dataset_name, dict_of_features, label_column
        )

    elif (isinstance(train_dataset, easy_rec.rec_torch.DictDataset) and isinstance(test_dataset, easy_rec.rec_torch.DictDataset)) \
        or (isinstance(train_dataset, easy_rec.rec_torch.Dataset) and test_dataset is None):
        
        dict_of_features = extract_features_from_easyrec_dataset(
            train_dataset, test_dataset, dataset_name, dict_of_features
        )

    else:
        raise ValueError("Unsupported dataset type. Please contact support to add compatibility.")

    return dict_of_features
