import os
import json

def create_dir_recommendation(dir_name:str="Real_Runs"):
    info_folder = os.path.join('..', 'out', 'exp', dir_name)
    if not os.path.exists('../../results'):
        os.mkdir('../../results')
    #iterate over all the files of this directory
    for single_file in os.listdir(info_folder):
        #load json file
        with open(os.path.join(info_folder, single_file), 'r') as f:
            data = json.load(f)
        model_name = data['model']['rec_model']['name']
        if not os.path.exists(os.path.join('..','results', model_name)):
            os.mkdir(os.path.join('..', 'results', model_name))
        dataset_name = data['data_params']['name']
        discard_percentage = int(data['data_params']['percentage'] * 100)
        batch_size = data['model']['loader_params']['batch_size']
        lr = data['model']['optimizer']['params']['lr']
        folder_name = f'{dataset_name}_discard_{discard_percentage}_{batch_size}_{lr}' 
        output_folder = os.path.join('../..', 'results', model_name, folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        predictions_folder= info_folder.replace('exp', 'log')
        os.system(f"cp {os.path.join(predictions_folder, single_file.replace('.json', ''),'emissions.csv')} {output_folder}")
        os.system(f"cp {os.path.join(predictions_folder, single_file.replace('.json', ''), 'train_flops.txt')} {output_folder}")
        os.system(f"cp -r {os.path.join(predictions_folder, single_file.replace('.json', ''), 'lightning_logs', 'version_0')} {output_folder}")

if __name__ == '__main__':
    create_dir_recommendation("Test_Real_Runs")