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
import pytorch_lightning as pl



class DataHandler():
    

    def __init__(self,base_path=None):
        self.base_path = base_path
        
        
    def load_yaml_from_drive(self, network_name, dataset_name):
        specific_path = f"{network_name}/{dataset_name}/results.yml"
        file_path = os.path.join(self.base_path, specific_path)

        if not os.path.exists(file_path):
           raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")
        
        try:
            with open(file_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
            return yaml_content
        except Exception as e:
           raise FileNotFoundError(f"Error reading the file '{file_path}': {e}")

    # Function to extract FLOPS from text file
    def extract_flops_from_text(self, network_name, dataset_name):
        specific_path = f"{network_name}/{dataset_name}/train_flops.txt"
        file_path = os.path.join(self.base_path, specific_path)

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
                raise ValueError(f"Error: Could not extract FLOPS from the file '{file_path}'")
        except Exception as e:
            raise FileNotFoundError(f"Error reading the file '{file_path}': {e}")

    def add_model_features(self, df, features_dict):
        # Loop over each row in the dataframe and assign num_layers and num_parameters based on network_name
        for index, row in df.iterrows():
            network = row['network_name']
            # Add num_layers and num_parameters from features_dict
            df.at[index, 'num_layers'] = features_dict['num_layers'].get(network, None)
            df.at[index, 'num_parameters'] = features_dict['num_parameters'].get(network, None)
        return df

    def extract_data(self, yaml_content):
        times_per_epoch = yaml_content['times_per_epoch']
        emissions_per_epoch = yaml_content['emissions_per_epoch']
        epoch_emissions_per_second = [e / t for e, t in zip(emissions_per_epoch, times_per_epoch)]

        return {
            #'num_params': yaml_content['num_params'],
            #'epochs_concluded': yaml_content['epochs_concluded'],
            'final_accuracy': yaml_content['accuracy'],
            #'total_emissions': yaml_content['total_emissions'],
            #'experiment_time': yaml_content['experiment_time'],
            #'total_emissions_per_second': yaml_content['total_emissions'] / yaml_content['experiment_time'],
            #'epoch_emissions_per_second': epoch_emissions_per_second,
            #'average_time_per_epoch': sum(times_per_epoch) / len(times_per_epoch),
            #'average_emissions_per_epoch': sum(emissions_per_epoch) / len(emissions_per_epoch),
            #'times_per_epoch': times_per_epoch,
            'emissions_per_epoch': emissions_per_epoch
        }

    def extract_custom_data(self, yaml_content, required_infos:list):
        results = {}
        for required in required_infos:
            results[required] = yaml_content[required]
            if required == 'emissions_per_epoch':
                results['epoch_emissions_per_second'] = [e / t for e, t in zip(results['emissions_per_epoch'], results['times_per_epoch'])]
        return results


    def expand_and_add_epochs(self, df):
        # Exploding both emissions_per_epoch and times_per_epoch columns
        df_expanded = df.explode(['emissions_per_epoch']).reset_index(drop=True) #, 'times_per_epoch'

        # Adding epoch column
        df_expanded['epoch'] = df_expanded.groupby('network_name').cumcount() + 1

        return df_expanded

    def expand_dataframe(self, dataset_list, network_list):

        dataframes = {}

        for dataset in dataset_list:
            data = []
            for network in network_list:
                yaml_content = self.load_yaml_from_drive(network, dataset)
                if yaml_content:
                    network_data = self.extract_data(yaml_content)
                    network_data['network_name'] = network
                    data.append(network_data)

            df = pd.DataFrame(data)
            dataframes[dataset] = df

        expanded_dataframes = {}
        for dataset_name, df in dataframes.items():
            df_expanded = self.expand_and_add_epochs(df)
            expanded_dataframes[dataset_name] = df_expanded

        combined_df = pd.concat(expanded_dataframes.values(), keys=expanded_dataframes.keys(), names=['dataset_name', 'row_id'])
        combined_df = combined_df.reset_index(level='dataset_name').reset_index(drop=True)
        return combined_df


def simulate_validation_accuracy(epochs, final_accuracy, noise_factor=0.02):  ### FATTO DA CHATGPT
    # Starting validation accuracy is typically low
    start_accuracy = 0.1 * final_accuracy  # Start at 10% of the final accuracy

    # Simulate accuracy progression using a logistic growth model
    epoch_range = np.arange(1, epochs + 1)
    growth_rate = 0.1  # Adjust this rate to control how fast accuracy grows
    validation_accuracy = start_accuracy + (final_accuracy - start_accuracy) * (1 - np.exp(-growth_rate * epoch_range))

    # Add stochastic noise to the validation accuracy
    noise = np.random.normal(0, noise_factor, size=epochs)
    validation_accuracy_with_noise = validation_accuracy + noise

    # Ensure accuracy is within the bounds [0, final_accuracy]
    validation_accuracy_with_noise = np.clip(validation_accuracy_with_noise, 0, final_accuracy)

    return validation_accuracy_with_noise


def split_data(features, target_val_acc, target_emissions, test_size=0.2, random_state=42):
    combined_targets = pd.concat([target_val_acc, target_emissions], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(features, combined_targets, test_size=test_size, random_state=random_state)

    y_train_val_acc, y_train_emissions = Y_train.iloc[:, 0], Y_train.iloc[:, 1]
    y_test_val_acc, y_test_emissions = Y_test.iloc[:, 0], Y_test.iloc[:, 1]

    return X_train, X_test, Y_train, Y_test, y_train_val_acc, y_test_val_acc, y_train_emissions, y_test_emissions


def preprocess_data(X_train, X_test, Y_train, Y_test, batch_size=16, fit_scalers=True, scalers=None):
    X_train_np, X_test_np = X_train.to_numpy(), X_test.to_numpy()
    Y_train_np, Y_test_np = Y_train.to_numpy(), Y_test.to_numpy()
    if fit_scalers:
        scaler_X, scaler_Y1, scaler_Y2 = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train_np)
        X_test_scaled = scaler_X.transform(X_test_np)

        Y_train_scaled_1 = scaler_Y1.fit_transform(Y_train_np[:, 0].reshape(-1, 1))
        Y_train_scaled_2 = scaler_Y2.fit_transform(Y_train_np[:, 1].reshape(-1, 1))

        Y_test_scaled_1 = scaler_Y1.transform(Y_test_np[:, 0].reshape(-1, 1))
        Y_test_scaled_2 = scaler_Y2.transform(Y_test_np[:, 1].reshape(-1, 1))

        scalers = {'scaler_X': scaler_X, 'scaler_Y1': scaler_Y1, 'scaler_Y2': scaler_Y2}
    else:
        scaler_X, scaler_Y1, scaler_Y2 = scalers['scaler_X'], scalers['scaler_Y1'], scalers['scaler_Y2']

        X_train_scaled = scaler_X.transform(X_train_np)
        X_test_scaled = scaler_X.transform(X_test_np)

        Y_train_scaled_1 = scaler_Y1.transform(Y_train_np[:, 0].reshape(-1, 1))
        Y_train_scaled_2 = scaler_Y2.transform(Y_train_np[:, 1].reshape(-1, 1))

        Y_test_scaled_1 = scaler_Y1.transform(Y_test_np[:, 0].reshape(-1, 1))
        Y_test_scaled_2 = scaler_Y2.transform(Y_test_np[:, 1].reshape(-1, 1))

    Y_train_scaled = np.hstack((Y_train_scaled_1, Y_train_scaled_2))
    Y_test_scaled = np.hstack((Y_test_scaled_1, Y_test_scaled_2))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, X_train_tensor, test_loader, scalers if fit_scalers else None

def get_df_final(model_info, combined_df, data_handler):
    model_info_df = pd.DataFrame.from_dict(model_info, orient='index', columns=['channels', 'perc_samples', 'classes', 'data_num', 'img_size']) # 'batch_size', 'l_r'])
    model_info_df = model_info_df.apply(pd.to_numeric)


    combined_df_with_info = combined_df.merge(model_info_df, left_on='dataset_name', right_index=True, how='left')
    combined_df_with_info['emissions_per_epoch'] = pd.to_numeric(combined_df_with_info['emissions_per_epoch'], errors='coerce')

    feature_path = 'features'
    feature_to_model_path = os.path.join(feature_path, 'features_to_models_dict.pkl')

    with open(feature_to_model_path, 'rb') as file:
        features_dict = pickle.load(file)
    combined_df_with_info = data_handler.add_model_features(combined_df_with_info, features_dict)

    print("The shape is.. ", combined_df_with_info.shape)

    #################### STARTING THE SIMULATION ####################

    validation_accuracy_list = []

    for (dataset_name, network_name), group in combined_df_with_info.groupby(['dataset_name', 'network_name']):
        final_accuracy = group['final_accuracy'].iloc[0]
        epochs = len(group)
        simulated_validation_accuracy = simulate_validation_accuracy(epochs, final_accuracy)


        validation_accuracy_list.extend(simulated_validation_accuracy)

    combined_df_with_info['val_acc'] = validation_accuracy_list

    ##### in the future put this in a function #####
    df_final=combined_df_with_info.copy()
    df_final['dataset_name'] = df_final['dataset_name'].apply(lambda x: x.split('_')[0])
    df_final = pd.get_dummies(df_final, columns=['network_name', 'dataset_name'], drop_first=False)
    df_final = df_final.drop(columns=['final_accuracy'])
    bool_columns = [col for col in df_final.columns if col.startswith("network_name_")]
    df_final[bool_columns] = df_final[bool_columns].astype(int)
    return df_final

def generate_input_combinations(new_input, model_space, epoch_space, features_dict):
    """Generate input combinations of model names, epochs, and new input features."""
    data_list = []
    for M in model_space:
        for e in epoch_space:
            input_features = new_input.copy()
            input_features.update({model_name: 1 if model_name == M else 0 for model_name in model_space})
            input_features['epoch'] = e
            model_key = M.replace('network_name_', '')

            input_features.update({feature_name: features_dict.get(feature_name, {}).get(model_key, None)
                                   for feature_name in ['num_layers', 'num_parameters']})

            data_list.append(input_features)

    return pd.DataFrame(data_list, columns=['epoch'] + list(new_input.keys()) +
                        ['num_layers', 'num_parameters'] + model_space)
#######################################################
###################################################################################

# class TransformerModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_heads=1, num_layers=2):
#         super(TransformerModel, self).__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(input_size, hidden_size)
#         self.fc_out = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.transformer_encoder(x)
#         x = self.fc(x)
#         x = self.dropout(x)
#         x = self.fc_out(x)

#         performance = torch.sigmoid(x[:, 0])
#         emissions = torch.sigmoid(x[:, 1])

#         return torch.stack([performance, emissions], dim=1)


# class EarlyStopping:
#     def __init__(self, patience=5, min_delta=0):

#         self.patience = patience
#         self.min_delta = min_delta
#         self.best_loss = None
#         self.counter = 0
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True


# def train_model(
#     model,
#     train_loader,
#     test_loader,
#     criterion,
#     optimizer,
#     scheduler=None,
#     num_epochs=200,
#     patience=100
# ):
#     early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
#     best_val_loss = float("inf")

#     train_losses, val_losses = [], []
#     train_mses, val_mses = [], []

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         mse_train = 0.0
#         total_samples = 0

#         for inputs, targets in train_loader:
#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             batch_mse = F.mse_loss(outputs, targets, reduction='sum').item()
#             mse_train += batch_mse
#             total_samples += targets.size(0)

#         average_train_loss = running_loss / len(train_loader)
#         average_train_mse = mse_train / total_samples

#         train_losses.append(average_train_loss)
#         train_mses.append(average_train_mse)

#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Train MSE: {average_train_mse:.4f}')

#         val_loss, val_mse = evaluate_model(model, test_loader, criterion)

#         val_losses.append(val_loss)
#         val_mses.append(val_mse)

#         early_stopping(val_loss)
#         if early_stopping.early_stop:
#             print(f"Early stopping at epoch {epoch+1}")
#             break

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), 'best_model.pth')
#             print(f"Saving best model at epoch {epoch+1}, Val Loss: {val_loss:.4f}")

#         if scheduler:
#             scheduler.step()

#     return train_losses, val_losses, train_mses, val_mses

# def evaluate_model(model, test_loader, criterion):
#     model.eval()
#     test_loss = 0.0
#     mse_test = 0.0
#     total_samples = 0

#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             test_loss += loss.item()  # Sum up the test loss

#             batch_mse = F.mse_loss(outputs, targets, reduction='sum').item()
#             mse_test += batch_mse
#             total_samples += targets.size(0)  # Total number of samples

#             # for i in range(min(2, len(outputs))):
#             #     print(f"Prediction: {outputs[i].cpu().numpy()}, Target: {targets[i].cpu().numpy()}")

#     average_test_loss = test_loss / len(test_loader)
#     average_test_mse = mse_test / total_samples

#     model.train()  # Set the model back to training mode
#     return average_test_loss, average_test_mse

# def plot_loss_and_mse(train_losses, val_losses, train_mses, val_mses):
#     epochs = range(1, len(train_losses) + 1)

#     plt.figure(figsize=(14, 6))

#     # Plot Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_losses, label='Training Loss')
#     plt.plot(epochs, val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()

#     # Plot MSE
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_mses, label='Training MSE')
#     plt.plot(epochs, val_mses, label='Validation MSE')
#     plt.xlabel('Epochs')
#     plt.ylabel('MSE')
#     plt.title('Training and Validation MSE')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()
#     plt.savefig('loss_mse_plot.png')





# TO DISCUSS
# 1. input size, hidden size, output size in the predict.py è "strano"
# 2. aggiunto relu in forward e decidere loss con vit + optmizer
# 3. early stopping in predict.py
# 4. mettere loggers e che tipo? bisogna usare tensorboard?

class TransformerModel(pl.LightningModule):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int, 
                 num_heads: int = 1, 
                 num_layers: int = 2):
        """
        Initializes the TransformerModel.

        Parameters:
        -----------
        input_size : int
            Input feature size.
        hidden_size : int
            Hidden layer size.
        output_size : int
            Output feature size.
        num_heads : int, optional
            Number of attention heads (default is 1).
        num_layers : int, optional
            Number of transformer encoder layers (default is 2).
        """
        super(TransformerModel, self).__init__()

        # Transformer Encoder Layer with attention heads
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Fully connected layers
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Dropout and ReLU activation
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters:
        -----------
        x : Tensor
            Input tensor with shape (sequence_length, batch_size, input_size).

        Returns:
        --------
        Tensor:
            Output tensor with shape (batch_size, 2), where the first value represents
            performance and the second value represents emissions.
        """
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = self.relu(x)  # added by Pippo
        x = self.dropout(x)
        x = self.fc_out(x)

        performance = torch.sigmoid(x[:, 0])
        emissions = torch.sigmoid(x[:, 1])

        return torch.stack([performance, emissions], dim=1)

     def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Parameters:
        -----------
        batch : tuple
            A tuple containing (input_tensor, target_tensor).
        batch_idx : int
            Index of the batch.

        Returns:
        --------
        Tensor:
            The loss value for the current batch.
        """
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = F.binary_cross_entropy(y_hat, y)  # change according to ViT
        
        # Calculate MSE
        mse_train = F.mse_loss(y_hat, y)

        # Log the loss and MSE
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse_train, on_step=True, on_epoch=True)

        # Append losses to lists for tracking
        self.train_losses.append(loss.item())
        self.train_mses.append(mse_train.item())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Parameters:
        -----------
        batch : tuple
            A tuple containing (input_tensor, target_tensor).
        batch_idx : int
            Index of the batch.

        Returns:
        --------
        None
        """
        x, y = batch
        y_hat = self(x)

        # Calculate loss
        loss = F.binary_cross_entropy(y_hat, y)

        # Calculate MSE
        mse_val = F.mse_loss(y_hat, y)

        # Log the validation loss and MSE
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse_val, on_step=True, on_epoch=True)

        # Append losses to lists for tracking
        self.val_losses.append(loss.item())
        self.val_mses.append(mse_val.item())

        # Early stopping
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.save_best_model()

    def save_best_model(self):
        """
        Save the best model state.
        """
        torch.save(self.state_dict(), 'best_model.pth')
        print("Best model saved.")

    def configure_optimizers(self):
        """
        Configures the optimizer.

        Returns:
        --------
        Optimizer:
            The optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


####################################### stuff for online predictor #######################################


# Updating theta function
def update_theta(q_theta, input_data, target_real, optimizer, criterion):
    print("Calling update_theta...")
    q_theta.train()

    optimizer.zero_grad()
    output = q_theta(input_data)
    P_pred, E_pred = output[:, 0], output[:, 1]
    print(f"P_pred in train mod: {P_pred}, E_pred in train mode: {E_pred}")
    print(f"Target_real in train mod: {target_real}")
    loss = criterion(output, target_real)
    loss.backward()
    optimizer.step()

    return loss.item()


def get_model_name(row):
    """Function to get the model name from network_name_XX columns"""
    for col in network_columns:
        if row[col] == 1:
            return col.replace('network_name_', '')  # Extract model name
    return 'Unknown'