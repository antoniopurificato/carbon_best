import os
import json
import shutil

def create_dir_recommendation():
    # Define the base info folder
    info_folder = os.path.join('D:\\results', 'out','exp', 'Real_Runs')

    # Ensure the base results folder exists
    results_base = 'D:\\results'
    if not os.path.exists(results_base):
        os.mkdir(results_base)

    # Iterate over all the files in the info folder
    if not os.path.exists(info_folder):
        raise FileNotFoundError(f"The directory {info_folder} does not exist.")
    
    for single_file in os.listdir(info_folder):
        # Load JSON file
        with open(os.path.join(info_folder, single_file), 'r') as f:
            data = json.load(f)

        # Extract parameters from the JSON data
        model_name = data['model']['rec_model']['name']
        dataset_name = data['data_params']['name']
        discard_percentage = int(data['data_params']['percentage'] * 100)
        batch_size = data['model']['loader_params']['batch_size']
        lr = data['model']['optimizer']['params']['lr']

        # Create model-specific folder in results
        model_folder = os.path.join(results_base, model_name)
        os.makedirs(model_folder, exist_ok=True)

        # Create output folder for this specific configuration
        folder_name = f'{dataset_name}_discard_{discard_percentage}_{batch_size}_{lr}'
        output_folder = os.path.join(model_folder, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Define the predictions folder
        predictions_folder = info_folder.replace('exp', 'log')

        # Copy relevant files
        files_to_copy = [
            'emissions.csv',
            'emission_eco2ai.csv',
            'train_flops.txt',
        ]
        for file_name in files_to_copy:
            src = os.path.join(predictions_folder, single_file.replace('.json', ''), file_name)
            dst = os.path.join(output_folder, file_name)
            try:
                shutil.copy(src, dst)
            except FileNotFoundError:
                print(f"File {src} not found. Skipping.")

        # Copy logs directory
        log_src = os.path.join(predictions_folder, single_file.replace('.json', ''), 'lightning_logs', 'version_0')
        log_dst = os.path.join(output_folder, 'version_0')
        try:
            shutil.copytree(log_src, log_dst)
        except FileNotFoundError:
            print(f"Directory {log_src} not found. Skipping.")

create_dir_recommendation()
