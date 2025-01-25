import os
import json
from copy import deepcopy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
import time
import codecarbon
from codecarbon import EmissionsTracker


class EmissionsTrackingCallback(pl.Callback):
    """
    A callback for tracking the emissions and time spent during training epochs using CodeCarbon.

    This callback logs the carbon emissions and training time for each epoch in a specified experiment folder. 
    It tracks the energy consumption of the training process and saves it into a CSV file for further analysis.

    Args:
        exp_id (str): The unique identifier for the experiment.
        exp_name (str): The name of the experiment to organize output logs.
    """

    def __init__(self, exp_id: str, exp_name: str):
        """
        Initializes the emissions tracking callback with experiment ID and name.

        Args:
            exp_id (str): The unique identifier for the experiment.
            exp_name (str): The name of the experiment for output directory organization.
        """
        # Store experiment details
        self.emissions_per_epoch = []  # List to store emissions per epoch
        self.times_per_epoch = []  # List to store time spent per epoch
        self.exp_name = exp_name  # Experiment name for directory structure
        self.exp_id = exp_id  # Experiment identifier

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Callback that gets triggered at the start of each training epoch.

        Initializes and starts the emissions tracker to monitor power usage during the epoch.
        Also, records the start time of the epoch for time tracking.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning model.
        """
        # Initialize the emissions tracker
        self.tracker = EmissionsTracker(
            log_level="critical",  # Set log level to avoid excessive logging
            tracking_mode="process",  # Track emissions for this specific process
            output_dir=f"../out/log/{self.exp_name}/{self.exp_id}",  # Directory to store the logs
            measure_power_secs=30,  # Frequency of power measurements in seconds
            api_call_interval=4,  # Interval for API calls in seconds
            allow_multiple_runs = True,  # Allow multiple runs in the same process
            experiment_id=self.exp_id,  # Set the experiment ID for output
            output_file='emissions.csv'  # Filename for emissions log
        )
        self.tracker.start()  # Start emissions tracking
        self.start_time = time.time()  # Record start time for epoch

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Callback that gets triggered at the end of each training epoch.

        Records the training time and emissions for the epoch, and appends them to respective lists.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning model.
        """
        # Calculate time spent in the current epoch
        epoch_time = time.time() - self.start_time
        self.times_per_epoch.append(epoch_time)  # Log epoch time

        # Stop emissions tracking and record the result
        latest_emission = self.tracker.stop()  # Stop emissions tracking and fetch data
        self.emissions_per_epoch.append(latest_emission)  # Log emissions for this epoch

        # Optionally, store epoch data in CSV or output for analysis
        epoch = trainer.current_epoch  # Get the current epoch
        # Optionally, print or log the epoch data for monitoring:
        # print(f"Epoch {epoch}: Time = {epoch_time:.2f}s, Emissions = {latest_emission:.2f}gCO2eq")
