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
import json
import pickle
import inspect
import re
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import copy
import matplotlib.pyplot as plt

from src.utils import *
from src.predictor.utils import *
from src.predictor.predictor import *


new_input = {'channels': 3, 'perc_samples': 1.0, 'classes': 3, 'data_num': 15000, 'img_size': 1024}
gamma = 0.2 #threshold performance minima
beta=0.5 #trade-off accuracy-emissioni (beta= importanza accuracy, 1-beta= importanza emissioni)

eta=0.001

dataset_list = ["cifar10_discard_100_16_0.0001760439131968857"]
network_list = get_models()

network_list.remove('vgg16')
network_list.remove('squeezenet')
network_list.remove('efficientnet')
network_list.remove('vit')

epoch_space = list(range(1, 15))
model_space = [
    'network_name_alexnet', 'network_name_efficientnet_b0',
    'network_name_mobilenet_v2', 'network_name_resnet101',
    'network_name_resnet18', 'network_name_squeezenet1_0',
    'network_name_swin_t', 'network_name_vgg16',
    'network_name_vit_b_16']


model_info = {
    'cifar10_20': (3, 1, 10, 50000, 1024),# 64, 0.001),
    'mnist_20': (1, 1, 10, 60000, 784),# 64 0.001),
    'cifar10_reduced-0.1_20': (3, 0.1, 10, 5000, 1024),# 64, 0.001),
    'cifar10_classes-123_20': (3, 1, 3, 15000, 1024)# 64, 0.001)
}

# STEP 0: PREPARE INPUT
with open(os.path.join(os.getcwd(), 'features', 'features_to_models_dict.pkl'), 'rb') as file:    
    features_dict = pickle.load(file)
df_input = generate_input_combinations(new_input, model_space, epoch_space, features_dict)
data_handler = DataHandler(base_path='results')
combined_df = data_handler.expand_dataframe(dataset_list, network_list)
df_final = get_df_final(model_info, combined_df, data_handler)
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

X_train, X_test, Y_train, Y_test, y_train_val_acc, y_test_val_acc, y_train_emissions, y_test_emissions = split_data(
    features, target_val_acc, target_emissions)
train_loader, X_train_tensor, test_loader, scalers = preprocess_data(X_train, X_test, Y_train, Y_test)

q_theta = TransformerModel(input_size=10, hidden_size=64, output_size=2)
# Scale the input features using the pre-fitted scaler

df_input = generate_input_combinations(new_input, model_space, epoch_space, features_dict)
X_scaled = scalers['scaler_X'].transform(df_input.values)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# STEP 1: Make predictions using q_theta
q_theta.eval()
with torch.no_grad():
    outputs = q_theta(X_tensor)

P_pred = outputs[:, 0].cpu().numpy()
E_pred = outputs[:, 1].cpu().numpy()

P_pred_original = scalers['scaler_Y1'].inverse_transform(P_pred.reshape(-1, 1)).flatten()
E_pred_original = scalers['scaler_Y2'].inverse_transform(E_pred.reshape(-1, 1)).flatten()

df_input['P_pred'] = P_pred_original
df_input['E_pred'] = E_pred_original

# STEP 2: Filter the dataframe based on the accuracy constraint (gamma)
df_filtered = df_input[df_input['P_pred'] >= gamma].copy()

if df_filtered.empty:
    print("No solutions meet the user's accuracy constraint.")
else:
    # Step 3: Identify (M*, e*) based on predictions and criteria
    optimizer = MultiCriteriaOptimizer(beta=0.5)
    df_sorted = optimizer.rank_and_sort_solutions_df(df_filtered, 'P_pred', 'E_pred')

    # Step 3.2: Rank test_only_df_filtered using real values (val_acc and emissions_per_epoch)
    test_only_df_filtered = test_only_df[test_only_df['val_acc'] >= gamma].copy()
    test_only_df_ranked = optimizer.rank_and_sort_solutions_df(test_only_df_filtered, 'val_acc', 'emissions_per_epoch')

    # Step 3.3: Find the best solution in test_only_df_filtered
    best_solution_found = False
    for idx, row in df_sorted.iterrows():
        best_model_indicator = row[model_space]
        M_star = best_model_indicator.idxmax()  # Best model in df_sorted
        e_star = row['epoch']  # Best epoch in df_sorted
        print(f"Checking candidate {idx + 1}: Model {M_star}, Epoch {e_star}: P_pred: {row['P_pred']}, E_pred: {row['E_pred']}")

        matching_row = test_only_df_filtered[(test_only_df_filtered[M_star] == 1) & (test_only_df_filtered['epoch'] == e_star)]
        if not matching_row.empty:
            best_solution_found = True
            print(f"Found matching row for model {M_star} and epoch {e_star} in test_only_df_filtered.")
            P_real = matching_row['val_acc'].values[0]  # Real validation accuracy
            E_real = matching_row['emissions_per_epoch'].values[0]  # Real emissions per epoch

            predicted_combination = test_only_df_ranked[
                (test_only_df_ranked[M_star] == 1) & (test_only_df_ranked['epoch'] == e_star)
            ]
            if not predicted_combination.empty:
                predicted_combination_rank = predicted_combination.index[0] + 1
                print(f"Model-epoch combination (M*={M_star}, e*={e_star}) ranked {predicted_combination_rank} out of {len(test_only_df_ranked)}.")

            break

    # STEP 4: Update theta based on the best solution found
    if best_solution_found:
        input_data = df_input.loc[(df_input[M_star] == 1) & (df_input['epoch'] <= e_star)].drop(columns=['P_pred', 'E_pred']).values
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        matching_rows = test_only_df_filtered[(test_only_df_filtered[M_star] == 1) & (test_only_df_filtered['epoch'] <= e_star)]
        P_real_values = matching_rows['val_acc'].values  # Real validation accuracies
        E_real_values = matching_rows['emissions_per_epoch'].values  # Real emissions per epoch
        print(f"P_real_values: {P_real_values}, E_real_values: {E_real_values}")

        target_real = torch.tensor(list(zip(P_real_values, E_real_values)), dtype=torch.float32)

        q_theta.train()
        optimizer = torch.optim.Adam(q_theta.parameters(), lr=eta)
        optimizer.zero_grad()

        output_pred = q_theta(input_tensor)
        print(f"P_pred in train mod: {output_pred[0,0]}, E_pred in train mode: {output_pred[0,1]}")
        criterion = nn.MSELoss()
        loss = criterion(output_pred, target_real)

        loss.backward()
        optimizer.step()

        print(f"Loss after updating theta: {loss.item()}")


################## compare results #########################
#display all columns
minetwork_columns = [col for col in df_sorted.columns if col.startswith('network_name_')]
df_sorted['Model'] = df_sorted[network_columns].idxmax(axis=1).str.replace('network_name_', '')

network_columns = [col for col in test_only_df_ranked.columns if col.startswith('network_name_')]
test_only_df_ranked['Model'] = test_only_df_ranked[network_columns].idxmax(axis=1).str.replace('network_name_', '')

network_columns = [col for col in test_only_df.columns if col.startswith('network_name_')]
test_only_df['Model'] = test_only_df[network_columns].idxmax(axis=1).str.replace('network_name_', '')

#pd.set_option('display.max_rows', None)
results_predicted=df_sorted[['Model','epoch', 'P_pred', 'E_pred', 'NB Score','KS Proportion', 'CP Score', 'NB rank', 'KS rank','CP rank', 'Borda rank']]
results_real=test_only_df_ranked[['Model','epoch', 'val_acc', 'emissions_per_epoch', 'NB Score','KS Proportion', 'CP Score', 'NB rank', 'KS rank','CP rank', 'Borda rank']]
merged_results = pd.merge(
    results_predicted,          # DataFrame with predicted values
    test_only_df[['Model', 'epoch', 'val_acc', 'emissions_per_epoch']],  # DataFrame with real values (only Model, epoch, val_acc, and emissions_per_epoch)
    on=['Model', 'epoch'],      # Merge on 'Model' and 'epoch'
    how='left'                  # Use 'left' join to keep all rows from df_sorted
)
pd.save_csv(merged_results, 'merged_results.csv')

################# aggregation ########################
max_P_true= test_only_df_ranked['val_acc'].max()
solution_bestP_true= test_only_df_ranked[test_only_df_ranked['val_acc']==max_P_true]

max_balance_true= test_only_df_ranked['Borda points'].max()
solution_bestBalance_true=test_only_df_ranked[test_only_df_ranked['Borda points']==max_balance_true]

max_P_pred=df_sorted['P_pred'].max()
solution_bestP_pred=df_sorted[df_sorted['P_pred']==max_P_pred]

max_balance_pred=df_sorted['Borda points'].max()
solution_bestBalance_pred=df_sorted[df_sorted['Borda points']==max_balance_pred]

############################ FINAL STUFF ########################################

network_columns = [col for col in test_only_df_ranked.columns if 'network_name_' in col]

solution_bestP_true_model = get_model_name(solution_bestP_true.iloc[0])
solution_bestBalance_true_model = get_model_name(solution_bestBalance_true.iloc[0])
solution_bestP_pred_model = get_model_name(solution_bestP_pred.iloc[0])
solution_bestBalance_pred_model = get_model_name(solution_bestBalance_pred.iloc[0])

# Create a dictionary to hold the solutions for each category
solution_dict = {
    'Solution Criteria': ['Max P_true', 'Max Balance_true', 'Max P_pred', 'Max Balance_pred'],
    'Model': [
        solution_bestP_true_model,
        solution_bestBalance_true_model,
        solution_bestP_pred_model,
        solution_bestBalance_pred_model
    ],
    'Epoch': [
        solution_bestP_true['epoch'].values[0],
        solution_bestBalance_true['epoch'].values[0],
        solution_bestP_pred['epoch'].values[0],
        solution_bestBalance_pred['epoch'].values[0]
    ],
    'P_true': [
        solution_bestP_true['val_acc'].values[0],
        solution_bestBalance_true['val_acc'].values[0],
        '-',  # Not applicable for predicted solutions
        '-'
    ],
    'E_true': [
        solution_bestP_true['emissions_per_epoch'].values[0],
        solution_bestBalance_true['emissions_per_epoch'].values[0],
        '-',  # Not applicable for predicted solutions
        '-'
    ],
    'Borda Points_true': [
        solution_bestP_true['Borda points'].values[0],
        solution_bestBalance_true['Borda points'].values[0],
        '-',  # Not applicable for predicted solutions
        '-'
    ],
    'P_pred': [
        '-',  # Not applicable for true solutions
        '-',
        solution_bestP_pred['P_pred'].values[0],
        solution_bestBalance_pred['P_pred'].values[0]
    ],
    'E_pred': [
        '-',  # Not applicable for true solutions
        '-',
        solution_bestP_pred['E_pred'].values[0],
        solution_bestBalance_pred['E_pred'].values[0]
    ],
    'Borda Points_pred': [
        '-',  # Not applicable for true solutions
        '-',
        solution_bestP_pred['Borda points'].values[0],
        solution_bestBalance_pred['Borda points'].values[0]
    ]
}

# Convert the dictionary to a dataframe
summary_df = pd.DataFrame(solution_dict)
pd.save_csv(summary_df, 'summary_df.csv')

print("FINITO")