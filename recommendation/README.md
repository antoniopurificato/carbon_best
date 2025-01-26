To use the library for our experiments, follow the next steps:

- If you are not located in this directory, enter:

```
cd recommendation
```

- Create the conda environment, with the required packages.

```
conda env create -f recsys.yaml
```

```
conda activate recsys
```

```
conda install -c conda-forge mpi4py mpich
```

- Download the data.

```
cd easy_rec
```

```
bash download_data.sh
```

To run an experiment:

```
cd ../ntb
```

```
python3 main_multi.py --lr [LEARNING_RATE_VALUES] --bs [BATCH_SIZE_VALUES] --ds [DATASET_NAME] --discard_percentage [DISCARD_PERCENTAGE] --model [MODEL_NAME]
```

The [ITEM] notation is used when you can insert more than 1 value for the corresponding ITEM.

If you run the experiments without adding arguments, it will create the entire knowledge base, as described in the paper (it could take months to complete all the exps).

We tried to do our best to sync our experiments with a repository provided by other researchers.
