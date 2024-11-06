import time
import os
import csv
import eco2ai
import torch 
from copy import deepcopy
from transformers import TrainerCallback
from codecarbon import EmissionsTracker
from transformers import (
    TrainerCallback,
    Conv1D
)

class EmissionsTrackingCallback_HF(TrainerCallback):
    def __init__(self, exp_name):
        self.emissions_per_epoch = []
        self.times_per_epoch = []
        self.exp_name = exp_name
        self.start_time = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # Start the emissions tracker at the beginning of training
        self.tracker = EmissionsTracker(
            log_level="critical",
            tracking_mode="process",
            output_dir=self.exp_name,
            measure_power_secs=30,
            api_call_interval=4,
            experiment_id=self.exp_name,
        )
        self.tracker.start()
        self.start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        # At the end of each epoch, record time and emissions
        epoch_time = time.time() - self.start_time
        self.times_per_epoch.append(epoch_time)
        self.start_time = time.time()  # Reset the start time for the next epoch
        
        # Stop and restart the tracker to measure per-epoch emissions
        # latest_emission = self.tracker.final_emissions_data
        # self.emissions_per_epoch.append(latest_emission)
        epoch = state.epoch
        #print(f"Epoch {state.epoch} ended. Time: {epoch_time:.2f}s, Emissions: {latest_emission['emissions']:.4f} CO2eq.")

    def on_train_end(self, args, state, control, **kwargs):
        # Stop the emissions tracker after training ends
        latest_emission = self.tracker.stop()
        self.emissions_per_epoch.append(latest_emission)
        # total_emissions = sum(e['emissions'] for e in self.emissions_per_epoch)
        #print(f"Training ended. Total emissions: {total_emissions:.4f} CO2eq.")



class SecondTrackerCallback_HF(TrainerCallback):
    def __init__(self, exp_name):
        self.emissions_per_epoch = []
        self.times_per_epoch = []
        self.exp_name = exp_name
        self.start_time = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # Start the emissions tracker at the beginning of training
        self.tracker = eco2ai.Tracker(
            project_name="Env footprint",
            file_name=f"{self.exp_name}/emission_eco2ai.csv",
            cpu_processes="current",
            measure_period=30,
            ignore_warnings=True,
            alpha_2_code="IT",  # Italy's country code for location-based emissions
        )
        self.tracker.start()
        self.start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        # At the end of each epoch, record time and emissions
        epoch_time = time.time() - self.start_time
        self.times_per_epoch.append(epoch_time)
        self.start_time = time.time()  # Reset start time for the next epoch

        # Log emissions for the current epoch
        # latest_emission = self.tracker.final_emissions_data
        # self.emissions_per_epoch.append(latest_emission)
        epoch = state.epoch
        # print(f"Epoch {state.epoch} ended. Time: {epoch_time:.2f}s, Emissions: {latest_emission['emissions']:.4f} CO2eq.")

    def on_train_end(self, args, state, control, **kwargs):
        # Stop the emissions tracker at the end of training
        emissions = self.tracker.stop()
        self.emissions_per_epoch.append(emissions)
        # total_emissions = sum(e['emissions'] for e in self.emissions_per_epoch)
        # print(f"Training ended. Total emissions: {total_emissions:.4f} CO2eq.")




def obtain_modules(model):
    layer_names = []
        
        # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing 

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    layer_names = list(set(layer_names))
    # delete the first element of the list
    del layer_names[0]
    return layer_names


class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer, save_path: str) -> None:
        super().__init__()
        self._trainer = trainer
        self.save_path = save_path
        self.csv_file = os.path.join(self.save_path, "training_log.csv")
        
        # Ensure the save path directory exists
        os.makedirs(self.save_path, exist_ok=True)
        
        # Initialize the CSV file with a header if it does not exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Value", "Metric_Name"])
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            
            # Perform evaluation on the training dataset
            eval_results = self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            
            # Extract accuracy from evaluation results (assumendo che l'accuratezza sia nel campo 'train_accuracy')
            accuracy = eval_results.get("train_accuracy", None)
            f1_micro = eval_results.get("train_f1_micro", None)
            f1_macro = eval_results.get("train_f1_macro", None)

            eval_results = self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="val")

            val_accuracy = eval_results.get('val_accuracy', None)
            val_f1_micro = eval_results.get("val_f1_micro", None)
            val_f1_macro = eval_results.get("val_f1_macro", None)
            
            # Append the epoch and accuracy to the CSV file
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([state.epoch, accuracy, "train_accuracy"])
                writer.writerow([state.epoch, f1_micro, "train_f1_micro"])
                writer.writerow([state.epoch, f1_macro, "train_f1_macro"])
                writer.writerow([state.epoch, val_accuracy, "val_accuracy"])
                writer.writerow([state.epoch, val_f1_micro, "val_f1_micro"])
                writer.writerow([state.epoch, val_f1_macro, "val_f1_macro"])

            
            return control_copy