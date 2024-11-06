import os
import yaml
import pandas as pd
import torch.nn.functional as F
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pickle
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from src.utils import *
from src.predictor.utils import *
from src.predictor.predictor import *


data_handler = DataHandler(base_path='results')
network_list = get_models()

network_list.remove('vgg16')
network_list.remove('squeezenet')
network_list.remove('efficientnet')
network_list.remove('vit')

##creare una funzione che itera su modelli e datasets e cerca gli esperimenti che sono in comune per tutti


#["resnet18", "resnet101", "alexnet", "vgg16", "squeezenet1_0", "efficientnet_b0", "vit_b_16", "mobilenet_v2", "swin_t"]
dataset_list = ["cifar10_discard_100_16_0.0001760439131968857"]

#ADDING INFOS ABOUT THE DATASET
model_info = {
    'cifar10_20': (3, 1, 10, 50000, 1024),
    'mnist_20': (1, 1, 10, 60000, 784),
    'cifar10_reduced-0.1_20': (3, 0.1, 10, 5000, 1024),
    'cifar10_classes-123_20': (3, 1, 3, 15000, 1024)
}

##togliere model info hardcoded, fa cagare


combined_df = data_handler.expand_dataframe(dataset_list, network_list)
df_final = get_df_final(model_info, combined_df, data_handler)
##### in the future put this in a function #####
test_only_df = df_final[
    (df_final['channels'] == 3) &
    (df_final['perc_samples'] == 1.0) &
    (df_final['classes'] == 3) &
    (df_final['data_num'] == 15000) &
    (df_final['img_size'] == 1024)
]


df_final_for_train = df_final[
    ~(
        (df_final['channels'] == 3) &
        (df_final['perc_samples'] == 1.0) &
        (df_final['classes'] == 3) &
        (df_final['data_num'] == 15000) &
        (df_final['img_size'] == 1024)
    )
]

columns_to_drop = lambda df: [col for col in df.columns if col.startswith('dataset_name_')]

df_final_for_train = df_final_for_train.drop(columns=columns_to_drop(df_final_for_train))
test_only_df = test_only_df.drop(columns=columns_to_drop(test_only_df))

features = df_final_for_train.drop(columns=['val_acc', 'emissions_per_epoch'])
target_val_acc = df_final_for_train['val_acc']
target_emissions = df_final_for_train['emissions_per_epoch']
#######################################################

X_train, X_test, Y_train, Y_test, y_train_val_acc, y_test_val_acc, y_train_emissions, y_test_emissions = split_data(
    features, target_val_acc, target_emissions)

train_loader, X_train_tensor, test_loader, scalers = preprocess_data(X_train, X_test, Y_train, Y_test)

print("Shapes are: ", X_train.shape, test_only_df.shape)



####################### Training ##########################
print("Training the model...")

input_size = X_train.shape[1]  # n. features in X_train
hidden_size = 128
output_size = Y_train.shape[1]  # performance (P') and impact (E')

model = TransformerModel(input_size=10, hidden_size=64, output_size=2)  # check this sizes
# decide how to plot the results and the logger to use

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=100,
    verbose=True,
    mode='min',
    stopping_threshold=0.001
)

trainer = pl.Trainer(
    max_epochs=200,
    callbacks=[early_stopping],
)

trainer.fit(model, train_loader, test_loader)
