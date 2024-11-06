import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import os
from pathlib import Path

from src.utils import get_models
from src.predictor.second_utils import extract_flops_from_text
    
class ArchitectureDataset(Dataset):
    def __init__(self, model_dict:dict, data_dict:dict):
        self.model_dict = model_dict
        self.data_dict = data_dict
        self.data_names = {'cifar10' : 'CIFAR10', 'cifar100' : 'CIFAR100',
        'mnist' : 'MNIST', 'food101' : 'FashionMNIST'}
        self.model_names = {'vit' : 'vit_b_16', 'efficientnet' : 'efficientnet_b0',
        'squeezenet' : 'squeezenet1_0', 'resnet18' : 'resnet18',
        'alexnet' : 'alexnet', 'vgg16' : 'vgg16'}
        self.model_dict = self.process_dict(model_dict)
        self.data_dict = self.process_dict(data_dict)
        self.entire_data = self.obtain_entire_dataset()

    def __len__(self):
        return len(self.entire_data)

    def __getitem__(self, idx):
        key = list(self.entire_data.keys())[idx]
        value = self.entire_data[key]

        # processed_value = self.process_dict(value)
        # processed_value['model_name']= key

        # dict_info = self.process_dict(self.data_dict[self.dataset_name])
        # processed_value.update(dict_info)
        #return processed_value
        return value

    def process_dict(self, single_feature):
        if isinstance(single_feature, dict):
            return {k: self.process_dict(v) for k, v in single_feature.items()}
        elif isinstance(single_feature, list):
            return [self.process_dict(v) for v in single_feature]
        # elif isinstance(single_feature, (int, float)):
        #     return torch.tensor(single_feature)
        else:
            return single_feature

    def obtain_precomputed_results(self, model_name:str, dataset_name:str,
                                   exp_name:str='discard_50_16_4.27022004702574e-05'):
        
        NOT_FOUND = False
        folder_name = os.path.join('results', model_name, f'{dataset_name.lower()}_{exp_name}')
        try:
            metrics = pd.read_csv(f'{folder_name}/version_0/metrics.csv')['val_acc'].dropna().values.tolist()
        except FileNotFoundError:
            try:
                metrics = pd.read_csv(f'{folder_name}/version_1/metrics.csv')['val_acc'].dropna().values.tolist()
            except FileNotFoundError:
                NOT_FOUND = True
                print(f'No metrics found for {model_name} on {dataset_name} with {exp_name}')
        if not NOT_FOUND:
            emissions_codecarbon = pd.read_csv(f'{folder_name}/emissions.csv')['emissions'].values.tolist()
            emissions_eco2ai = pd.read_csv(f'{folder_name}/emission_eco2ai.csv')['CO2_emissions(kg)'].values.tolist()
            return torch.tensor(metrics), torch.tensor(emissions_codecarbon), torch.tensor(emissions_eco2ai)
        else:
            return torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

    def obtain_entire_dataset(self):
        
        final_data = {}
        all_models = get_models()
        results_folder = Path('results')
        for s in results_folder.iterdir():
            if str(s).replace('results/', '') in all_models:
                for subpath in s.iterdir():
                    model_name = str(subpath).replace('results/', '').split('_')[0].split('/')[0]
                    dataset_name = str(subpath).replace('results/', '').split('_')[0].split('/')[1]
                    discard_percentage = int(str(subpath).replace('results/', '').split('_')[2])
                    batch_size = int(str(subpath).replace('results/', '').split('_')[3])
                    learning_rate = float(str(subpath).replace('results/', '').split('_')[4])
                    keys = (model_name, dataset_name, discard_percentage, batch_size, learning_rate)
                    final_data[keys] = {}

                    #final_data[keys]['model_name'] = str(subpath).replace('results/', '').split('_')[0].split('/')[0]
                    #final_data[keys]['dataset_name'] = str(subpath).replace('results/', '').split('_')[0].split('/')[1]
                    try:
                        final_data[keys]['FLOPS'] = extract_flops_from_text(subpath)
                    except FileNotFoundError:
                        final_data[keys]['FLOPS'] = -1
                    final_data[keys]['discard_percentage'] = int(str(subpath).replace('results/', '').split('_')[2])
                    final_data[keys]['batch_size'] = int(str(subpath).replace('results/', '').split('_')[3])
                    final_data[keys]['learning_rate'] = float(str(subpath).replace('results/', '').split('_')[4])

                    res = self.obtain_precomputed_results(model_name = keys[0], dataset_name = keys[1],
                                                    exp_name = f'discard_{keys[2]}_{keys[3]}_{keys[4]}')
                    final_data[keys]['val_acc'] = res[0]
                    final_data[keys]['emissions_codecarbon'] = res[1]
                    final_data[keys]['emissions_eco2ai'] = res[2]

                    final_data[keys].update(self.data_dict[self.data_names[keys[1]]])
                    final_data[keys].update(self.model_dict[self.model_names[keys[0]]])

                    final_data[keys] = self.process_dict(final_data[keys])
                    final_data[keys]['num_sample_per_class'] = final_data[keys]['num_train_examples'] * 
                    del final_data[keys]['num_test_examples']
        return final_data


def remove_features(data_dict: dict, features_to_remove: list):
    if not isinstance(features_to_remove, list):
        raise ValueError("features_to_remove must be a list")
    for model_key, model_value in data_dict.items():
        keys_to_remove = [key for key in model_value.keys() if key in features_to_remove]
        for key in keys_to_remove:
            del model_value[key]
    return data_dict
    
if __name__ == '__main__':
    with open('features/datasets_to_features_dict.pkl', 'rb') as handle:
        datasets_to_features = pickle.load(handle)
    with open('features/models_to_features_dict.pkl', 'rb') as handle:
        models_to_features = pickle.load(handle)
    #dataset = ArchitectureDataset(remove_features(models_to_features, ['num_parameters']), datasets_to_features, 'CIFAR10')
    dataset = ArchitectureDataset(models_to_features, datasets_to_features)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(dataset.entire_data[('vit', 'cifar10', 50, 16, 4.27022004702574e-05)])