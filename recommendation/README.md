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
bash download_data.bash
```

To run an experiment:

```
cd ../ntb
```

```
python3 main_multi.py
```

If you run the experiments, it will create the entire knowledge base, as described in the paper (it could take months to complete all the exps).

If you want to select only specific configurations, you have to modify the specific config files!

- Dataset $\rightarrow$ `cfg/data_params/data_cfg/name`
- Batch size $\rightarrow$ `cfg/model/loader_params/loader_params_cfg/£batch_size`
- Discard percentage $\rightarrow$ `cfg/data_params/data_cfg/percentage`
- Learning rate $\rightarrow$ `cfg/model/trainer_params/trainer_params_cfg/lr`
- Model $\rightarrow$ `cfg/model/model/rec_model`

We tried to do our best to sync our experiments with a repository provided by other researchers.
