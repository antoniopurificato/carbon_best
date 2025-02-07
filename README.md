# One Search Fits All: Pareto-Optimal Eco-Friendly Model Selection

The environmental impact of Artificial Intelligence (AI) is emerging as a significant global concern, particularly regarding model training. In this paper, we introduce **GREEN** (Guided Recommendations of Energy-Efficient Networks), a novel, inference-time approach for recommending Pareto-optimal AI model configurations that optimize validation performance and energy consumption across diverse AI domains and tasks. Our approach directly addresses the limitations of current eco-efficient neural architecture search methods, which are often restricted to specific architectures or tasks. Central to this work is EcoTaskSet, a dataset comprising training dynamics from over **1767** experiments across computer vision, natural language processing, and recommendation systems using both widely used and cutting-edge architectures. Leveraging this dataset and a prediction model, our approach demonstrates effectiveness in selecting the best model configuration based on user preferences. Experimental results show that our method successfully identifies energy-efficient configurations while ensuring competitive performance.


This project is structured into several key components, each responsible for a different part of the workflow, such as data processing, model training, result extraction, and utility functions.


## EcoTaskSet

To download the knowledge base we have created, [click here](https://drive.google.com/drive/folders/1lXdSsW2FRU331bpGWOsXrcg-Bp3Px4Pi?usp=sharing). 

If you want to recreate the knowledge base from scratch, follow the next steps:

- Create a conda environment:

```
conda create -n eco python=3.10
```

- Activate the conda environment:

```
conda activate eco
```

- Download and unzip this repository. Since it is anonymous, it can not be cloned.

```
unzip carbon_best-81DE.zip
```

- Ensure that you are in the main folder. Create the `results` folder:

```
mkdir results
```

- Install the required libraries:

```
pip3 install -r requirements.txt
```

- To create the *vision* knowledge base use the following command:
```
python3 train_vision.py --dataset [DATASETS_NAMES] --model [MODEL_NAMES] --epochs [NUM_EPOCHS] --gpu_id ID_OF_THE_GPU --seed SEED --bs [BATCH_SIZE_VALUES] --discard_percentage [DISCARD_PERCENTAGE_VALUES] --lr [LR_VALUES]
```

- To create the *textual* knowledge base use the following command, **remember to insert a valid HuggingFace key!**:

```
python3 train_text.py --huggingface_key KEY --dataset [DATASETS_NAMES] --model [MODEL_NAMES] --epochs [NUM_EPOCHS] --seed SEED --discard_percentage [DISCARD_PERCENTAGE_VALUES] --lr [LR_VALUES]
```

  Warning! If you have multiple GPUs HuggingFace is having some issues. The only solution we were able to find out was to put `CUDA_VISIBLE_DEVICES=2` before the previous command, i.e., `CUDA_VISIBLE_DEVICES=2 python3 train_text.py ...`. 

- To create the *recommendation systems* knowledge base enter in the `recommendation` folder and follow the steps of the `README.md` file you can find in that folder.

The [ITEM] notation is used when you can insert more than 1 value for the corresponding ITEM.

If you run the experiments without adding arguments, it will create the entire knowledge base, as described in the paper (it could take months to complete all the exps).

## GREEN 

0. **Libraries Installation**:
Before starting, you have to follow the procedures of **EcoTaskSet** and install the required libraries via the command `pip3 install -r requirements.txt`.

1. **Data Preparation**:
The project starts by loading datasets using `prepare_data.py` and configuring the model settings via `predictor_config.yaml`. From the base folder run:

```
python3 -m src.predictor.prepare_data
```

If you do not have the data from the `EcoTaskSet` dataset, this script will automatically download it in the `src\KB\` folder.

2. **Model Training and Prediction**:

The model is trained using the Transformer architecture (`temporal_transformer.py`) through `train_predictor.py`. Predictions are saved for evaluation and further analysis. To run the code, use:

```
python3 -m src.predictor.train_predictor
```

3. **Results Extraction**:

After training is complete and checkpoints have been saved, run:
```
python3 -m src.predictor.extract_and_save_results
```

This script will automatically download the pretrained `.ckpt.` files if you did not train the architecture.

3b. **HPO**:

If you want to perform HPO (not mandatory), run:

```
python3 -m hpo_predictor --wandb_project PROJECT_NAME_OF_THE_SWEEP --wandb_key WANDB_KEY
```

4. **Multi-Objective Optimization (MOO)**:

The MOO module evaluates the performance of models using Pareto-optimal metrics and rankings.

- Run the initial extraction of results:
```
python3 -m src.MOO.1_extract_results
```
This step aggregates and summarizes the prediction results.

- To analyze metrics related to model predictions, run:
```
python3 -m src.MOO.2a_elaborate_MAE_results
```

- To compute metrics related to the Pareto front, run:
```
python3 -m src.MOO.2b_elaborate_pareto_metrics
```

- To rank the solutions and compute Pareto-based metrics, run:
```
python3 -m src.MOO.3_get_pareto_ranking_metrics
```

- To generate the exact table of results presented in the paper, run:
```
python3 -m src.MOO.3a_elaborate_ndcg_sova
```

All the results are saved in `src\results_csv`.

By following the above steps, you will be able to replicate the results presented in the paper and generate the required tables, metrics, and summaries.
