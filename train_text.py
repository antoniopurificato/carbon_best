import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import evaluate
import numpy as np
import random
import eco2ai
import time
import pandas as pd
import yaml


from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    pipeline
)


from deepspeed.profiling.flops_profiler import FlopsProfiler
from codecarbon import EmissionsTracker
from peft import LoraConfig, get_peft_model, TaskType
from peft import prepare_model_for_kbit_training

from src.utils import *
from src.text_model import *


def remove_percentage_of_samples(dataset, percentage):
    # Calcolare il numero di campioni da rimuovere
    num_samples_to_remove = int(len(dataset) * percentage / 100)
    
    # Generare una lista di indici casuali da tenere (non rimuovere)
    indices_to_keep = random.sample(range(len(dataset)), len(dataset) - num_samples_to_remove)
    
    # Selezionare solo i campioni da mantenere
    reduced_dataset = dataset.select(indices_to_keep)
    
    return reduced_dataset


def map_labels_column(dataset):  # poi verrà fatto che da solo lui mi da le colonne e le label
    if dataset=='google/boolq':
        train_dataset = load_dataset(dataset, split="train")
        test_dataset = load_dataset(dataset, split="validation")
        return train_dataset, test_dataset, 2, ['question', 'answer', 'passage']
    elif dataset=='dair-ai/emotion':
        train_dataset = load_dataset(dataset, split="train")
        test_dataset = load_dataset(dataset, split="test")
        return train_dataset, test_dataset, 6, ['text', 'label']
    elif dataset=='stanfordnlp/imdb':
        train_dataset = load_dataset(dataset, split="train")
        test_dataset = load_dataset(dataset, split="test")
        return train_dataset, test_dataset, 2, ['text', 'label']
    # elif dataset=="allenai/ai2_arc ARC-Challenge":
    #     dataset.split(' ')
    #     train_dataset = load_dataset(dataset.split(' ')[0], dataset.split(' ')[1], split="auxilary_train")
    #     test_dataset = load_dataset(dataset.split(' ')[0], dataset.split(' ')[1], split="test")
    #     return train_dataset, test_dataset, 4, ['question', 'choices', 'answerKey']
    else:
        raise ValueError("Dataset not found")

# Funzione di preprocessamento dei dati
def preprocess_function(examples, tokenizer, columns_to_use):
    if 'passage' in columns_to_use:
        inputs = [" Question: " + q for q in examples['question']]
        #  inputs = ["Passage: " + p + " Question: " + q for p, q in zip(examples['passage'], examples['question'])]

        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        # Convert 'yes'/'no' to integers 1/0 for labels
        labels = [1 if a else 0 for a in examples['answer']]
        model_inputs["labels"] = labels
    else:
        model_inputs = tokenizer(examples[columns_to_use[0]], max_length=512, truncation=True, padding="max_length")
        
        # Add labels directly, ensuring they are integers
        labels = examples[columns_to_use[1]]
        model_inputs["labels"] = labels

    return model_inputs


# Funzione per il calcolo delle metriche
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load('accuracy')
    f1 = evaluate.load('f1')
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_micro = f1.compute(predictions=predictions, references=labels, average='micro')
    f1_macro = f1.compute(predictions=predictions, references=labels, average='macro')
    return {"accuracy": acc["accuracy"], "f1_micro": f1_micro["f1"], "f1_macro": f1_macro["f1"]}




# Funzione di training del modello
def train_model(model_name, 
                train_dataset, 
                test_dataset, 
                num_labels, 
                columns_to_use, 
                perc_value, 
                dataset_name,
                learning_rate=2e-4,
                num_epochs=2):

    
    app = dataset_name.replace("/", "_")
    app2 = model_name.replace("/", "_")

    emissions_res = {}
    exp_name = f"{app}_discard_{perc_value}_{str(num_epochs)}_{str(learning_rate)}"
    folder_path = "results/" + str(app2)
    exp_name = os.path.join(folder_path, exp_name)
    if not os.path.isfile(os.path.join(exp_name,'results.yml')):
        print(f"Experiment: {exp_name}")
        if not os.path.isdir(exp_name):
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            os.mkdir(exp_name)

        # Carica tokenizer e modello
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
        if 'Mistral' in model_name or 'phi' in model_name or 'llama' in model_name:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Preprocessa i dataset
        tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer, columns_to_use), batched=True)
        tokenized_test = test_dataset.map(lambda x: preprocess_function(x, tokenizer, columns_to_use), batched=True)

        # Carica il modello
        if 'bert' in model_name:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                trust_remote_code=True,
                use_cache=False  # Disable cache to be compatible with gradient checkpointing

                # attn_implementation="flash_attention_2" 
            )        
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                use_cache=False  # Disable cache to be compatible with gradient checkpointing

                # attn_implementation="flash_attention_2" 
            )
            model = prepare_model_for_kbit_training(model)

            list_modules = obtain_modules(model)
            # Apply LoRA (Low-Rank Adaptation) to the model
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,   # Task type: sequence classification
                r=8,                          # LoRA attention dimension
                lora_alpha=16,                # LoRA scaling factor
                lora_dropout=0.1,             # LoRA dropout
                bias="none",                  # Keep bias
                target_modules=list_modules  # Choose target modules for LoRA
            )

            model.resize_token_embeddings(len(tokenizer))
            # Wrap your model with the LoRA config
            model = get_peft_model(model, lora_config)

        profiler = FlopsProfiler(model)
        profiler.output_dir = exp_name

        # Definisci gli argomenti per il training
        training_args = TrainingArguments(
            output_dir=exp_name,
            learning_rate=learning_rate,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,  # Cambia a True se vuoi inviare il modello a Hugging Face Hub
        )

        emissions_callback = EmissionsTrackingCallback_HF(exp_name)
        emissions_eco2ai_callback = SecondTrackerCallback_HF(exp_name)

        # Inizializza il Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[emissions_callback, emissions_eco2ai_callback],
        )
        trainer.add_callback(CustomCallback(trainer, save_path=f"{exp_name}/epochs_results"))
        # Allena il modello

        total_tracker = EmissionsTracker(
                            tracking_mode="process",
                            log_level="critical",
                            output_dir=exp_name,
                            measure_power_secs=30,
                            api_call_interval=4,
                            experiment_id=exp_name,
                        )
        eco2ai_tracker = eco2ai.Tracker(
                                project_name="Env footprint",
                                file_name=f"{exp_name}/emission_eco2ai.csv",
                                cpu_processes="current",
                                measure_period=30,
                                ignore_warnings=True,
                                alpha_2_code="IT",
                            )
        eco2ai_tracker.start()
        total_tracker.start()
        profiler.start_profile()

        experiment_start_time = time.time()
        trainer.train()

        experiment_end_time = time.time()
        experiment_time = experiment_end_time - experiment_start_time

        # Valuta il modello
        results = trainer.evaluate()

        # Salva i risultati
        total_emissions = total_tracker.stop()
        profiler.print_model_profile(
            output_file=f"{profiler.output_dir}/train_flops.txt"
        )
        eco2ai_tracker.stop()
        emissions_eco2ai = float(pd.read_csv(f"{exp_name}/emission_eco2ai.csv").iloc[0]['CO2_emissions(kg)'])
        profiler.stop_profile()
        emissions_res[exp_name] = {
                                "accuracy": results["eval_accuracy"],
                                "f1_micro": results["eval_f1_micro"],
                                "f1_macro": results["eval_f1_macro"],
                                "num_params": compute_model_params(
                                    trainer.model
                                ),
                                "flops": profiler.get_total_flops(), #check
                                "epochs_concluded": num_epochs,
                                "experiment_time": experiment_time,
                                "total_emissions": total_emissions,
                                "emissions_per_epoch": emissions_callback.emissions_per_epoch,
                                "times_per_epoch": emissions_callback.times_per_epoch,
                                "emissions_eco2ai": emissions_eco2ai,
                                "original_data_size" : int(len(train_dataset) / (100-perc_value)*100),
                                "modified_data_size" : len(train_dataset),
                            }

        with open(os.path.join(exp_name, "results.yml"), "w") as yaml_file:
            yaml.dump(
                emissions_res[exp_name], yaml_file, default_flow_style=False
            )
    else:
        print(f"Experiment {exp_name} already exists") 
        return None

    return results




def main():
    # Example Dataset: Replace with your desired dataset
    seed_everything(42)
    dataset_name = ["google/boolq", "dair-ai/emotion", "stanfordnlp/imdb"]

   
    # Define different model names
    model_names = [
        "roberta-base",
        "bert-base-uncased",
        'microsoft/phi-2',
        'meta-llama/Meta-Llama-3.1-8B',
        'mistralai/Mistral-7B-v0.3',
        "distilbert-base-uncased"
    ]


    # create a list of 10 random learning rates between 1e-7 and 1e-2
    learning_rate = [2e-2, 8e-2, 6e-3, 1e-3, 5e-4, 1e-7, 5e-6, 6e-5, 1e-6, 1e-4]

    # Train and evaluate each model
    for model_name in model_names:

        for datas in dataset_name:
            for samples_percentage in [0, 30, 50, 70, 90]:
                for lr in learning_rate:
                    # Load the dataset
                    # Prepare the dataset            
                    train_dataset, test_dataset, label, columns_to_use = map_labels_column(datas)
                    train_reduced_dataset = remove_percentage_of_samples(train_dataset, samples_percentage)


                    print(f"Training and evaluating model: {model_name}")



                    results = train_model(model_name, train_reduced_dataset, test_dataset, label, columns_to_use, samples_percentage, datas, learning_rate=lr, num_epochs=3)   


if __name__ == "__main__":
    seed_everything(42)
    
    main()

