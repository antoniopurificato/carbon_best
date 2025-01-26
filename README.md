# One Search Fits All: A Pareto-Optimal Approach to Environmentally Aware Model Selection

Code to reproduce the results of the paper.

## EcoTaskSet

In order to recreate the knowledge base, follow the next steps:

- Create a conda environment:

```
conda create -n eco python=3.10
```

- Activate the conda environment:

```
conda activate eco
```

- Download this repository. Since it is anonymous, it can not be cloned.

- Enter in the main directory.

```
cd carbon_best
```

- Create the `results` folder:

```
mkdir results
```

- Install the required libraries:

```
pip3 install -r requirements.txt
```

- To create the *vision* knowledge base use the following command:
```
python3 train_vision.py --dataset [DATASETS_NAMES] --model [MODEL_NAMES] --epochs [NUM_EPOCHS] --gpu_id ID_OF_THE_GPU --seed SEED --bs [BATCH_SIZE_VALUES] -- discard_percentage [DISCARD_PERCENTAGE_VALUES] --lr [LR_VALUES]
```

- To create the *textual* knowledge base use the following command, **remember to insert a valid HuggingFace key!**:

```
python3 train_text.py --huggingface_key KEY --dataset [DATASETS_NAMES] --model [MODEL_NAMES] --epochs [NUM_EPOCHS] --gpu_id ID_OF_THE_GPU --seed SEED -- discard_percentage [DISCARD_PERCENTAGE_VALUES] --lr [LR_VALUES]
```

  Warning! If you have multiple GPUs HuggingFace is having some issues. The only solution we were able to find out was to put `CUDA_VISIBLE_DEVICES=2` before the previous command, i.e., `CUDA_VISIBLE_DEVICES=2 python3 train_text.py ...`. 

- To create the *recommendation systems* knowledge base enter in the `recommendation` folder and follow the steps of the `README.md` file you can find in that folder.

The [ITEM] notation is used when you can insert more than 1 value for the corresponding ITEM.

If you run the experiments without adding arguments, it will create the entire knowledge base, as described in the paper (it could take months to complete all the exps).
