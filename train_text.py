import os
import evaluate
import numpy as np
import random
import time
import pandas as pd
import yaml
import argparse
import deepspeed
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    pipeline,
)
from deepspeed.profiling.flops_profiler import FlopsProfiler
from codecarbon import EmissionsTracker
from peft import LoraConfig, get_peft_model, TaskType
from peft import prepare_model_for_kbit_training
from huggingface import login

from src.utils.secondary_utils import *
from src.text_model import *


def remove_percentage_of_samples(dataset, validation, percentage):
    """
    Reduces the number of samples in a dataset and its validation set by a given percentage.

    Args:
        dataset (Dataset): The training dataset to reduce.
        validation (Dataset): The validation dataset to reduce.
        percentage (float): Percentage of samples to remove.

    Returns:
        tuple: Reduced training and validation datasets.
    """
    num_samples_to_remove = int(len(dataset) * percentage / 100)
    indices_to_keep = random.sample(
        range(len(dataset)), len(dataset) - num_samples_to_remove
    )
    reduced_dataset = dataset.select(indices_to_keep)

    val_samples_to_remove = int(len(validation) * percentage / 100)
    val_indices_to_keep = random.sample(
        range(len(validation)), len(validation) - val_samples_to_remove
    )
    reduced_val = validation.select(val_indices_to_keep)

    return reduced_dataset, reduced_val


def map_labels_column(dataset_name: str = "google/boolq"):
    """
    Maps dataset-specific fields to a unified format for training.

    Args:
        dataset_name (str): Name of the dataset to load and preprocess.

    Returns:
        tuple: Training, validation, and test datasets, number of labels, and columns to use.
    """
    if dataset_name == "google/boolq":
        train_val_dataset = load_dataset(dataset_name, split="train").train_test_split(
            test_size=0.3
        )
        train_dataset = train_val_dataset["train"]
        val_dataset = train_val_dataset["test"]
        test_dataset = load_dataset(dataset_name, split="validation")
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            2,
            ["question", "answer", "passage"],
        )

    elif dataset_name == "dair-ai/emotion":
        train_val_dataset = load_dataset(dataset_name, split="train").train_test_split(
            test_size=0.3
        )
        train_dataset = train_val_dataset["train"]
        val_dataset = train_val_dataset["test"]
        test_dataset = load_dataset(dataset_name, split="test")
        return train_dataset, val_dataset, test_dataset, 6, ["text", "label"]

    elif dataset_name == "stanfordnlp/imdb":
        train_val_dataset = load_dataset(dataset_name, split="train").train_test_split(
            test_size=0.3
        )
        train_dataset = train_val_dataset["train"]
        val_dataset = train_val_dataset["test"]
        test_dataset = load_dataset(dataset_name, split="test")
        return train_dataset, val_dataset, test_dataset, 2, ["text", "label"]

    elif dataset_name == "cornell-movie-review-data/rotten_tomatoes":
        train_val_dataset = load_dataset(dataset_name, split="train").train_test_split(
            test_size=0.3
        )
        train_dataset = train_val_dataset["train"]
        val_dataset = train_val_dataset["test"]
        test_dataset = load_dataset(dataset_name, split="test")
        return train_dataset, val_dataset, test_dataset, 2, ["text", "label"]

    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def preprocess_function(examples, tokenizer, columns_to_use):
    """
    Tokenizes the input text and prepares labels for training.

    Args:
        examples (dict): Batch of examples from the dataset.
        tokenizer (AutoTokenizer): Tokenizer to preprocess text.
        columns_to_use (list): List of column names to use for inputs and labels.

    Returns:
        dict: Tokenized inputs with labels.
    """
    if "passage" in columns_to_use:
        inputs = [" Question: " + q for q in examples["question"]]
        model_inputs = tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length"
        )

        # Convert 'yes'/'no' to integers 1/0 for labels
        labels = [1 if a else 0 for a in examples["answer"]]
        model_inputs["labels"] = labels
    else:
        model_inputs = tokenizer(
            examples[columns_to_use[0]],
            max_length=512,
            truncation=True,
            padding="max_length",
        )

        # Add labels directly, ensuring they are integers
        labels = examples[columns_to_use[1]]
        model_inputs["labels"] = labels

    return model_inputs


def compute_metrics(eval_pred):
    """
    Computes evaluation metrics such as accuracy.

    Args:
        eval_pred (tuple): Predictions and labels from the evaluation set.

    Returns:
        dict: Computed metrics.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = evaluate.load("accuracy")
    acc = accuracy.compute(predictions=predictions, references=labels)

    # Sanity check for predictions
    if (
        len(set(predictions)) <= 1  # All predictions are the same
        or predictions.shape[0] != labels.shape[0]  # Shape mismatch
    ):
        print(
            "Sanity check failed: Invalid predictions. Returning minimal metrics for f1 scores."
        )
        return {"accuracy": acc["accuracy"]}

    return {"accuracy": acc["accuracy"]}


def train_model(
    model_name: str,
    train_dataset,
    val_dataset,
    test_dataset,
    num_labels,
    columns_to_use,
    perc_value,
    dataset_name: str,
    learning_rate: float = 2e-4,
    num_epochs: int = 2,
    gpu_id: str = "0",
):
    """
    Trains a model using the provided datasets and configuration.

    Args:
        model_name (str): Name of the pretrained model.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
        num_labels (int): Number of output labels.
        columns_to_use (list): List of column names for inputs and labels.
        perc_value (float): Percentage of samples discarded during preprocessing.
        dataset_name (str): Name of the dataset.
        learning_rate (float): Learning rate for training.
        num_epochs (int): Number of training epochs.
        gpu_id (str): GPU ID to use for training.

    Returns:
        dict: Evaluation results.
    """
    # Prepare unique experiment directory structure
    dataset_name = dataset_name.replace("/", "_")
    model_name = model_name.replace("/", "_")

    emissions_res = {}
    exp_name = f"{dataset_name}_discard_{perc_value}_{str(learning_rate)}"
    folder_path = "results/" + str(model_name)
    exp_name = os.path.join(folder_path, exp_name)

    # Check if results already exist, skip if they do
    if not os.path.isfile(os.path.join(exp_name, "results.yml")):
        print(f"Experiment: {exp_name}")
        if not os.path.isdir(exp_name):
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)  # Create model folder if it doesn't exist
            os.mkdir(exp_name)  # Create experiment folder

        # Load tokenizer for the specified model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Add special tokens if necessary for certain models
        if "Mistral" in model_name or "phi" in model_name or "llama" in model_name:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Tokenize datasets
        tokenized_train = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer, columns_to_use), batched=True
        )
        tokenized_val = val_dataset.map(
            lambda x: preprocess_function(x, tokenizer, columns_to_use), batched=True
        )
        tokenized_test = test_dataset.map(
            lambda x: preprocess_function(x, tokenizer, columns_to_use), batched=True
        )

        # Load the appropriate model with the correct configuration
        if "bert" in model_name:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                trust_remote_code=True,
                use_cache=False,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True
                ),  # Enable 8-bit quantization
                use_cache=False,
            )
            model = prepare_model_for_kbit_training(
                model
            )  # Prepare the model for efficient training with LoRA

            # Identify target modules for LoRA adaptation
            list_modules = obtain_modules(model)

            # Apply Low-Rank Adaptation (LoRA) configuration
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,  # Task type: Sequence Classification
                r=8,  # LoRA attention dimension
                lora_alpha=16,  # Scaling factor
                lora_dropout=0.1,  # Dropout rate
                bias="none",  # Bias configuration
                target_modules=list_modules,  # Target layers for LoRA
            )

            model.resize_token_embeddings(
                len(tokenizer)
            )  # Resize token embeddings to match tokenizer
            model = get_peft_model(
                model, lora_config
            )  # Wrap model with LoRA configuration

        # Initialize FLOPs profiler to monitor model computational efficiency
        profiler = FlopsProfiler(model)
        profiler.output_dir = exp_name

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=exp_name,
            learning_rate=learning_rate,
            per_device_train_batch_size=1,  # Batch size for training
            per_device_eval_batch_size=1,  # Batch size for evaluation
            num_train_epochs=num_epochs,  # Number of epochs
            weight_decay=0.01,  # Weight decay for regularization
            evaluation_strategy="epoch",  # Evaluate at the end of each epoch
            save_strategy="epoch",  # Save model at the end of each epoch
            load_best_model_at_end=True,  # Load the best model after training
            push_to_hub=False,  # Do not push model to the Hugging Face hub
            report_to=None,  # Disable reporting to any external tools
        )

        # Initialize emissions tracker for environmental impact
        emissions_callback = EmissionsTrackingCallback_HF(exp_name)

        # Define the Trainer object for training and evaluation
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,  # Pass the metric computation function
            callbacks=[emissions_callback],  # Add emissions tracking callback
        )

        # Add custom callback for saving intermediate results
        trainer.add_callback(
            CustomCallback(trainer, save_path=f"{exp_name}/epochs_results")
        )

        # Track total emissions during the experiment
        total_tracker = EmissionsTracker(
            tracking_mode="process",
            log_level="critical",
            output_dir=exp_name,
            measure_power_secs=30,
            api_call_interval=4,
            experiment_id=exp_name,
        )

        # Start tracking emissions and FLOPs profiling
        total_tracker.start()
        profiler.start_profile()

        # Train the model
        experiment_start_time = time.time()
        trainer.train()

        # Measure experiment time
        experiment_end_time = time.time()
        experiment_time = experiment_end_time - experiment_start_time

        # Evaluate the model on the test dataset
        results = trainer.evaluate(tokenized_test)

        # Stop emissions tracking and save results
        total_emissions = total_tracker.stop()
        profiler.print_model_profile(
            output_file=f"{profiler.output_dir}/train_flops.txt"
        )
        profiler.stop_profile()

        # Save experiment results in a structured format
        emissions_res[exp_name] = {
            "accuracy": results["eval_accuracy"],
            "num_params": compute_model_params(trainer.model),
            "flops": profiler.get_total_flops(),
            "epochs_concluded": num_epochs,
            "experiment_time": experiment_time,
            "total_emissions": total_emissions,
            "emissions_per_epoch": emissions_callback.emissions_per_epoch,
            "times_per_epoch": emissions_callback.times_per_epoch,
            "original_data_size": int(len(train_dataset) / (100 - perc_value) * 100),
            "modified_data_size": len(train_dataset),
        }

        # Write the results to a YAML file
        with open(os.path.join(exp_name, "results.yml"), "w") as yaml_file:
            yaml.dump(emissions_res[exp_name], yaml_file, default_flow_style=False)
    else:
        # Skip training if results already exist
        print(f"Experiment {exp_name} already exists")
        return None

    return results


def main():
    """
    Main function to parse arguments and orchestrate the model training and evaluation.
    """
    parser = argparse.ArgumentParser()

    # Add dataset and model arguments
    parser.add_argument(
        "--dataset", default=get_all_datasets("text"), nargs="+", type=str
    )
    parser.add_argument("--model", default=get_models("text"), nargs="+", type=str)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--huggingface_key", type=str, default = "NA", help="HuggingFace key to download the models.")

    # Add arguments for discard percentage and learning rate
    parser.add_argument(
        "--discard_percentage",
        type=int,
        nargs="+",
        default=[70, 30, 0],
        help="Values of discard percentage",
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[10e-3, 10e-4, 10e-5],
        help="Learning rate values",
    )
    parser.add_argument("--seed", default=42, type=int)
    seed_everything(args.seed)

    args = parser.parse_args()
    gpu_id = args.gpu_id
    dataset_name = args.dataset
    model_names = args.model

    if args.huggingface_key == "NA":
        raise ValueError(f"Insert a valid HuggingFace key! {args.huggingface_key} is not valid!")
    login(args.huggingface_key)

    # Train models for all combinations of hyperparameters
    for samples_percentage in args.discard_percentage:
        for lr in args.lr:
            for datas in dataset_name:
                for model_name in model_names:
                    # Load datasets and preprocess
                    train_dataset, val_dataset, test_dataset, label, columns_to_use = (
                        map_labels_column(datas)
                    )
                    train_reduced_dataset, validation_reduced_dataset = (
                        remove_percentage_of_samples(
                            train_dataset, val_dataset, samples_percentage
                        )
                    )

                    # Train and evaluate the model
                    print(f"Training and evaluating model: {model_name}")
                    results = train_model(
                        model_name,
                        train_reduced_dataset,
                        validation_reduced_dataset,
                        test_dataset,
                        label,
                        columns_to_use,
                        samples_percentage,
                        datas,
                        learning_rate=lr,
                        num_epochs=5,
                        gpu_id=gpu_id,
                    )


if __name__ == "__main__":
    main()
