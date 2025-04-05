In order to perform NAS, follow the next steps:

Create necessary folders:

```
mkdir NAS/results_NAS && mkdir NAS/benchmarks/nasbench101 && mkdir NAS/benchmarks/nasbench101/dataset
```

Split the `data.json` dataset file into smaller chuncks to easily process it:

```
python -m NAS.benchmarks.divide_nasbench
```

If necessary, set up your config in `carbon_best/NAS/configs/predictor_config_NAS.yaml`. For `nasbench101` we used `label_len=108` of CIFAR-10.

Run the predictor:

```
python -m NAS.predictor.train_predictor_NAS
```

Extract the results:

```
python -m NAS.predictor.extract_and_save_results_NAS
```

Save Pareto fronts:

```
python -m NAS.MOO.get_pareto_fronts_NAS
```

Obtain the ranking:

```
python -m NAS.MOO.get_ranking_NAS
```

In order to perform a sanity check of the proposed approach:

Start training the predictor:

```
python -m src.sanity_check.train_predictor_sanity_check
```

Extract the predictions:

```
python -m src.predictor.extract_and_save_results --sanity_check
```

Before extracting results:

```
python -m src.MOO.1_extract_results --sanity_check
```

**Obtain results for the baseline**:

Start training the predictor:

```
python -m src.predictor.train_predictor
```

Extract the predictions:

```
python -m src.predictor.extract_and_save_results 
```


In order to get the results:

```
python -m src.sanity_check.diff_MAE_sanity_check
```

If you want to change some of the configs, you should go to `src/configs/predictor_config.yaml`.