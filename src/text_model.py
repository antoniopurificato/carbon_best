import time
import os
import csv
import torch
from copy import deepcopy
from transformers import TrainerCallback, Conv1D
from codecarbon import EmissionsTracker


class EmissionsTrackingCallback_HF(TrainerCallback):
    def __init__(self, exp_name):
        self.emissions_per_epoch = []
        self.times_per_epoch = []
        self.exp_name = exp_name
        self.start_time = 0
        self.tracker = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize the emissions tracker and start tracking."""
        self.tracker = EmissionsTracker(
            log_level="critical",
            tracking_mode="process",
            output_dir=self.exp_name,
            measure_power_secs=30,
            api_call_interval=4,
            experiment_id=self.exp_name,
            allow_multiple_runs=True,
        )
        self.tracker.start()
        self.start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Record the elapsed time for each epoch."""
        epoch_time = time.time() - self.start_time
        self.times_per_epoch.append(epoch_time)
        self.start_time = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        """Stop the emissions tracker after training and log emissions."""
        if self.tracker:
            latest_emission = self.tracker.stop()
            self.emissions_per_epoch.append(latest_emission)


def obtain_modules(model):
    """Get target layer names for LoRA or quantization."""
    layer_names = []

    for name, module in model.named_modules():
        if isinstance(
            module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)
        ):
            # Extract relevant part of the module name
            layer_name = ".".join(name.split(".")[4:]).split(".")[0]
            layer_names.append(layer_name)

    # Remove duplicates and exclude the first element
    return list(set(layer_names))[1:]


class CustomCallback(TrainerCallback):
    def __init__(self, trainer, save_path: str):
        super().__init__()
        self._trainer = trainer
        self.save_path = save_path
        self.csv_file = os.path.join(self.save_path, "training_log.csv")

        # Ensure the save path directory exists
        os.makedirs(self.save_path, exist_ok=True)

        # Initialize the CSV file with a header if it doesn't exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Value", "Metric_Name"])

    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate the model and log metrics to a CSV file."""
        if control.should_evaluate:
            # Evaluate training metrics
            train_results = self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            val_results = self._trainer.evaluate(
                eval_dataset=self._trainer.eval_dataset, metric_key_prefix="val"
            )

            # Extract key metrics
            metrics = {
                "train_accuracy": train_results.get("train_accuracy"),
                "train_f1_micro": train_results.get("train_f1_micro"),
                "train_f1_macro": train_results.get("train_f1_macro"),
                "val_accuracy": val_results.get("val_accuracy"),
                "val_f1_micro": val_results.get("val_f1_micro"),
                "val_f1_macro": val_results.get("val_f1_macro"),
            }

            # Append metrics to the CSV file
            with open(self.csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                for metric_name, value in metrics.items():
                    writer.writerow([state.epoch, value, metric_name])

            return deepcopy(control)
