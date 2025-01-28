#!/usr/bin/env python
# coding: utf-8

# # Preparation stuff

# ## Connect to Drive

# In[1]:



# In[2]:


connect_to_drive = False


# In[3]:


#Run command and authorize by popup --> other window
if connect_to_drive:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)


# ## Install packages

# In[4]:


if connect_to_drive:
    #Install FS code
    get_ipython().system('pip install  --upgrade --force-reinstall git+https://github.com/federicosiciliano/easy_lightning.git')

    get_ipython().system('pip install pytorch_lightning')


# ## IMPORTS

# In[5]:


#Put all imports here
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
#import pickle
import os
import sys
#import cv2
import csv
import deepspeed
import torch

from create_dir_recommendation import create_dir_recommendation


# ## Define paths

# In[6]:

deepspeed.init_distributed(dist_backend=None, distributed_port=29497)

#every path should start from the project folder:
project_folder = "../"
if connect_to_drive:
    project_folder = "/content/gdrive/Shareddrives/<SharedDriveName>" #Name of SharedDrive folder
    #project_folder = "/content/gdrive/MyDrive/<MyDriveName>" #Name of MyDrive folder

#Config folder should contain hyperparameters configurations
cfg_folder = os.path.join(project_folder,"cfg")

#Data folder should contain raw and preprocessed data
data_folder = os.path.join(project_folder,"data")
raw_data_folder = os.path.join(data_folder,"raw")
processed_data_folder = os.path.join(data_folder,"processed")

#Source folder should contain all the (essential) source code
source_folder = os.path.join(project_folder,"src")

#The out folder should contain all outputs: models, results, plots, etc.
out_folder = os.path.join(project_folder,"out")
img_folder = os.path.join(out_folder,"img")


# ## Import own code

# In[7]:


#To import from src:

#attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)

#import from src directory
# from src import ??? as additional_module
import easy_rec as additional_module #REMOVE THIS LINE IF IMPORTING OWN ADDITIONAL MODULE

import easy_exp, easy_rec, easy_torch #easy_data


# # MAIN

# ## Train

# ### Data

# In[7]:


cfg = easy_exp.cfg.load_configuration('config_rec')

# In[8]:


cfg["data_params"]["data_folder"] = raw_data_folder

# In[9]:


cfg

# In[10]:


for _ in cfg.sweep('data_params.name'):
    for _ in cfg.sweep("model.loader_params.batch_size"):
        for _ in cfg.sweep("data_params.percentage"):
            for lr in cfg.sweep("model.optimizer.params.lr"):

                #cfg["data_params"]["test_sizes"] = [cfg["data_params.dataset_params.val_size"],cfg["data_params.dataset_params.test_size"]]
                
                data, maps = easy_rec.data_generation_utils.preprocess_dataset(**cfg["data_params"])

                #TODO: save maps

                # In[11]:
                batch_size = cfg["model.loader_params"]["batch_size"]


                datasets = easy_rec.rec_torch.prepare_rec_datasets(data,**cfg["data_params"]["dataset_params"])
                num_users = np.max(list(maps["uid"].values()))

                print("Len of original dataset: ", easy_rec.carbon_best_utils.calculate_avg_length(datasets, num_users))
                datasets = easy_rec.carbon_best_utils.remove_samples_per_user(datasets, num_users, cfg["data_params"]["percentage"])
                print("Len of dataset after removal: ", easy_rec.carbon_best_utils.calculate_avg_length(datasets, num_users))


                collator_params = cfg["data_params"]["collator_params"].copy()

                collator_params["num_items"] = np.max(list(maps["sid"].values()))

                # In[ ]:


                collators = easy_rec.rec_torch.prepare_rec_collators(data, **collator_params)


                # In[12]:


                loaders = easy_rec.rec_torch.prepare_rec_data_loaders(datasets, **cfg["model"]["loader_params"], collate_fn=collators)


                # In[13]:

                rec_model_params = cfg["model"]["rec_model"].copy()
                rec_model_params["num_items"] = np.max(list(maps["sid"].values()))
                rec_model_params["num_users"] = np.max(list(maps["uid"].values()))
                rec_model_params["lookback"] = cfg["data_params"]["collator_params"]["lookback"]

                # In[14]:


                main_module = easy_rec.rec_torch.create_rec_model(**rec_model_params)

                # In[15]:


                exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
                print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)


                # In[16]:


                if exp_found: continue #TODO: make the notebook stop here if the experiment is already found


                # In[17]:


                trainer_params = easy_torch.preparation.prepare_experiment_id(cfg["model"]["trainer_params"], experiment_id)

                # Prepare callbacks and logger using the prepared trainer_params
                trainer_params["callbacks"] = easy_torch.preparation.prepare_callbacks(trainer_params)
                trainer_params["logger"] = easy_torch.preparation.prepare_logger(trainer_params)

                exp_namess = cfg["__exp__"]["name"]

                # eco2 =  easy_torch.callbacks.SecondTrackerCallback(experiment_id, exp_namess)
                codecar = easy_rec.callback_carbonbest.EmissionsTrackingCallback(experiment_id, exp_namess) #easy_torch.callbacks.EmissionsTrackingCallback(experiment_id, exp_namess)

                # trainer_params["callbacks"].append(eco2)
                trainer_params["callbacks"].append(codecar)
                
                # Prepare the trainer using the prepared trainer_params
                trainer = easy_torch.preparation.prepare_trainer(**trainer_params)
                

                model_params = cfg["model"].copy()

                model_params["loss"] = easy_torch.preparation.prepare_loss(cfg["model"]["loss"], easy_rec.losses)

                # Prepare the optimizer using configuration from cfg
                model_params["optimizer"] = easy_torch.preparation.prepare_optimizer(**cfg["model"]["optimizer"])

                # Prepare the metrics using configuration from cfg
                model_params["metrics"] = easy_torch.preparation.prepare_metrics(cfg["model"]["metrics"], easy_rec.metrics)

                # Create the model using main_module, loss, and optimizer
                model = easy_torch.process.create_model(main_module, **model_params)


                # In[18]:


                # Prepare the emission tracker using configuration from cfg
                tracker = easy_torch.preparation.prepare_emission_tracker(**cfg["model"]["emission_tracker"], experiment_id=experiment_id)

            #  eco2aitracker = easy_torch.preparation.prepare_eco2ai_tracker(**cfg["model"]["emission_tracker"], experiment_id=experiment_id)

                # In[19]:


                # Prepare the flops profiler using configuration from cfg
                profiler = easy_torch.preparation.prepare_flops_profiler(model=model, **cfg["model"]["flops_profiler"], experiment_id=experiment_id)


                # In[21]:


                # Train the model using the prepared trainer, model, and data loaders
                easy_torch.process.train_model(trainer, model, loaders, tracker=tracker, val_key=["val","test"], profiler=profiler) #, eco2aitracker=eco2aitracker)


                # In[22]:


                easy_torch.process.test_model(trainer, model, loaders, tracker=tracker, profiler=profiler) #, eco2aitracker=eco2aitracker)


                # In[23]:


                # Save experiment and print the current configuration
                #save_experiment_and_print_config(cfg)
                easy_exp.exp.save_experiment(cfg)

                # Print completion message
                print("Execution completed.")
                print("######################################################################")
                print()

                    # In[ ]:
create_dir_recommendation()
