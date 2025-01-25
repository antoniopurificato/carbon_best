from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import pickle
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import json
import os

from temporal_transformer import TransformerPredictor
from data_new import ArchitectureDataset
from src.utils.plot import *
from src.utils.main_utils_new import load_yaml_exp_folder

if __name__ == '__main__':
    with open(load_yaml_exp_folder()[0], 'rb') as handle:
        datasets_to_features = pickle.load(handle)
    with open(load_yaml_exp_folder()[1], 'rb') as handle:
        models_to_features = pickle.load(handle)

    # Initialize the full dataset
    full_dataset = ArchitectureDataset(models_to_features, datasets_to_features)

    test_data = full_dataset.filter_test_data(dataset_name='food101', discard_percentage=50)
    remaining_data = {key: value for key, value in full_dataset.valid_data.items() if key not in test_data}
    full_dataset.valid_data = remaining_data

    val_split_ratio = 0.20 # Proportion of data for validation
    train_size = int(len(full_dataset) * (1 - val_split_ratio))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = ArchitectureDataset(models_to_features, datasets_to_features)
    test_dataset.valid_data = test_data  # Assign the filtered test data

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Total exp: {len(test_dataset)+len(val_dataset)+len(train_dataset)}")

    # Initialize data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    num_features = 769 #1658 all  #1624 vision+text, 736 vision only #769 recsys+vision
    seq_len = 400
    num_targets = 2

    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",           
    dirpath="ckpt/",       
    filename="best_model_transf_noem",  
    save_top_k=1,                
    mode="min", 
    verbose=True
)

    # Define the EarlyStopping callback
    early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # Metric to monitor
    patience=15,          # Number of epochs with no improvement after which training will stop
    verbose=True,        # Print early stopping message
    mode="min"           # "min" because we want to minimize the validation loss
)
    
    model = TransformerPredictor(num_features=num_features, seq_len=seq_len, num_targets=num_targets)
    
    trainer= pl.Trainer(max_epochs=400, log_every_n_steps=10, callbacks=[early_stopping_callback, checkpoint_callback], logger=True)
    trainer.fit(model,train_dataloader, val_dataloader)
    
    best_model_path = checkpoint_callback.best_model_path  # Path to the best model
    print(f"Best model saved at: {best_model_path}")
          
    best_model = TransformerPredictor.load_from_checkpoint(best_model_path)
    best_model.eval() 

    test_predictions_direct=[]
    #test_predictions = []
    test_labels = []
    test_exps_plot=[]
    test_exps=[]
    split_exps=[]
    #test_uncertainties = []


    with torch.no_grad():  
        for batch in test_dataloader: 
            key, inputs, labels = batch

            key_str = "_".join(
                str(item[0]) if isinstance(item, tuple) else str(item.item()) if isinstance(item, torch.Tensor) else str(item)
                for item in key
            )
            
            test_exps.extend([key_str] * labels.size(1))
            print(f"key_str: '{key_str}'")
            
                # Split key_str into components and store them
            model_name, dataset_name, data_perc, bs, lr = key_str.split("_")
            split_exps.extend(
                    [{"model_name": model_name, "dataset_name": dataset_name, "data_perc": data_perc, "bs": bs, "lr": lr}]
                    * labels.size(1)
                )
        
            predictions_direct = best_model(inputs) 
            predictions_direct = predictions_direct[:100]
            labels = labels[:100]

            test_predictions_direct.append(predictions_direct.cpu())    
            test_labels.append(labels.cpu())
            test_exps_plot.append(key_str) 
        
            #mean_predictions, uncertainty_estimates = best_model.predict_with_uncertainty(inputs, n_samples=50)
            #test_predictions.append(mean_predictions.cpu())
            #test_uncertainties.append(uncertainty_estimates.cpu())



    test_predictions_direct= torch.cat(test_predictions_direct, dim=0)
    #test_predictions = torch.cat(test_predictions, dim=0)
    #test_uncertainties = torch.cat(test_uncertainties, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    test_predictions_direct_np = test_predictions_direct.numpy()
    #test_predictions_np = test_predictions.numpy()
    #test_uncertainties_np = test_uncertainties.numpy()
    test_labels_np = test_labels.numpy()

    # Flatten the data
    #num_samples = test_predictions_np.shape[0]
    #seq_len = test_predictions_np.shape[1]


    direct_predicted_ACC=test_predictions_direct_np[:, :, 0].flatten()
    #predicted_ACC = test_predictions_np[:, :, 0].flatten()
    #uncertainty_ACC = test_uncertainties_np[:, :, 0].flatten()
    direct_predicted_EM=test_predictions_direct_np[:, :, 1].flatten()
    #predicted_EM = test_predictions_np[:, :, 1].flatten()
    #uncertainty_EM = test_uncertainties_np[:, :, 1].flatten()

    true_ACC = test_labels_np[:, :, 0].flatten()
    true_EM = test_labels_np[:, :, 1].flatten()

    # Prepare the DataFrame
    data = {
        "true_ACC": true_ACC,
        #"predicted_ACC": predicted_ACC,
        "predicted_ACC":  direct_predicted_ACC,
        #"uncertainty_ACC": uncertainty_ACC,
        "true_EM": true_EM,
        #"predicted_EM": predicted_EM,
        "predicted_EM": direct_predicted_EM, 
        #"uncertainty_EM": uncertainty_EM,
    }

    flattened_exps = pd.DataFrame(split_exps)
    original_key_strs = pd.DataFrame({"key_str": test_exps})

    df = pd.concat([original_key_strs.reset_index(drop=True), flattened_exps.reset_index(drop=True), pd.DataFrame(data)], axis=1)
    df['epoch'] = df.groupby('key_str').cumcount() + 1


    # Save to CSV
    csv_path = "test_results.csv"
    df.to_csv(os.path.join("results_csv", csv_path), index=False)
    print(f"Test results saved to {csv_path}")


    print("Test results saved to test_results_dict.csv in CSV-like format")
    #PLOT
    pred_first = test_predictions_direct_np[:, :, 0]  
    pred_rest = test_predictions_direct_np[:, :, 1]  
    label_first = test_labels_np[:, :, 0]  
    label_rest = test_labels_np[:, :, 1]  

    mask_first = label_first != -1
    mask_rest = label_rest != -1
    

    diff_first = np.abs(pred_first[mask_first] - label_first[mask_first])
    diff_rest = np.abs(pred_rest[mask_rest] - label_rest[mask_rest])
   
    #std_pred_first = test_uncertainties_np[:, :, 0]      # Uncertainties for VAL_ACC
    #std_pred_rest = test_uncertainties_np[:, :, 1]       # Uncertainties for EMISSION


    # Compute the Mean Absolute Error (MAE) for each label
    mae_first = np.mean(diff_first)  # MAE for the first label
    mae_rest = np.mean(diff_rest)    # MAE for the second label

    #avg_uncertainty_first = np.mean(std_pred_first)
    #avg_uncertainty_rest = np.mean(std_pred_rest)

    # Print the results
    print(f"MAE for the VAL_ACC label: {mae_first}")
    #print(f"Average Uncertainty for VAL_ACC: {avg_uncertainty_first}")

    print(f"MAE for the EMISSION label: {mae_rest}")
    #print(f"Average Uncertainty for EMISSION: {avg_uncertainty_rest}")

    #uncertainty_diff_first=std_pred_first 
   # uncertainty_diff_rest=std_pred_rest

    n_samples= diff_first.shape[0]
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    plot_absolute_difference(n_samples, colors, diff_first, test_exps_plot)

    losses_df = pd.read_csv("results_csv/training_validation_losses.csv")
    plot_losses(losses_df) 
    
