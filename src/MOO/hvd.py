import json
import numpy as np
import os
import pandas as pd
from tabulate import tabulate


def prepare_front(pareto_set, acc_key, en_key):
    front = []
    for item in pareto_set:
        acc = item.get(acc_key, 0.0)
        en = item.get(en_key, 0.0)
        flipped_en = 1.0 - en
        front.append([acc, flipped_en])
    return np.array(front)

def manual_hypervolume(front, ref_point):
    is_pareto = np.ones(front.shape[0], dtype=bool)
    for i in range(front.shape[0]):
        for j in range(front.shape[0]):
            if i != j and np.all(front[j] >= front[i]):
                is_pareto[i] = False
                break
    filtered_front = front[is_pareto]
    sorted_front = filtered_front[filtered_front[:, 0].argsort()[::-1]]
    
    hypervolume = 0.0
    prev_y = ref_point[1]  
    
    for point in sorted_front:
        width = point[0] - ref_point[0]
        height = point[1] - prev_y
        hypervolume += width * height
        prev_y = point[1]
    
    return hypervolume

def compute_hvd_manual(true_front, predicted_front, ref_point: list = None):
    if ref_point is None:
        combined = np.vstack([true_front, predicted_front])
        
        ref_point = np.array([
            np.min(combined[:, 0]) - 1e-6,  # For ACC (maximized)
            np.min(combined[:, 1]) - 1e-6   # For 1-EN (maximized)
        ])
    else:
        ref_point = np.array(ref_point)

    hv_true = manual_hypervolume(true_front, ref_point)
    hv_pred = manual_hypervolume(predicted_front, ref_point)
    
    if hv_true == 0:
        return np.nan
    
    hvd = (hv_true - hv_pred) / hv_true
    return hvd


def filter_non_dominated(front):
    """Remove dominated points within a front (for 2D objectives)"""
    if len(front) == 0:
        return front

    sorted_front = sorted(front, key=lambda p: (p[0], -p[1]))
    filtered = []
    max_y = -np.inf
    for x, y in sorted_front:
        if y > max_y:  # Non-dominated
            filtered.append([x, y])
            max_y = y
    return np.array(filtered)


seeds = [42, 2025, 476]
test_datasets = ["foursquare_tky","rotten_tomatoes", "cifar10"]

# Dictionary to store hypervolumes for each configuration across seeds
ref_point = {"cifar10": [0.09, 0.8], "rotten_tomatoes": [0.47, 0.8], "foursquare_tky": [0.8, 0.8]}

hv_results = {}

for seed in seeds:
    file_name = f"pareto_results_{seed}.json"
    if not os.path.exists(file_name):
        print(f"File {file_name} not found. Skipping.")
        continue

    with open(file_name, "r") as f:
        data = json.load(f)

    for dataset in test_datasets:
        data_perc = ["100", "70", "30"] if dataset == 'cifar10' else ["0", "30", "70"]

        for d in data_perc:
            ground_truth = data[dataset][d]["ground_truth"]
            predicted = data[dataset][d]["predicted"]
            discard_percentage = 100 - int(d) if dataset == 'cifar10' else int(d)

            true_front = prepare_front(ground_truth, "true_ACC", "true_EN")
            pred_front = prepare_front(predicted, "true_ACC", "true_EN" )
            hvd = compute_hvd_manual(true_front, pred_front, ref_point[dataset])
            
            config_key = (dataset, discard_percentage)
            if config_key not in hv_results:
                hv_results[config_key] = {"diff": []}

            hv_results[config_key]["diff"].append(hvd)
 
records = []
for config, results in hv_results.items():
    dataset, discard_percentage = config
    hv_diff_mean = np.mean(results["diff"])
    hv_diff_std = np.std(results["diff"])

    records.append([
        dataset,
        discard_percentage,
        f'{hv_diff_mean:.4f} ± {hv_diff_std:.4f}'
    ])

records_overall = []

for config, results in hv_results.items():
    dataset, discard_percentage = config
    hv_diff_mean = np.mean(results["diff"])
    hv_diff_std = np.std(results["diff"])

    records_overall.append([
        dataset,
        discard_percentage,
        f'{hv_diff_mean:.5f} ± {hv_diff_std:.5f}'
    ])

df_overall = pd.DataFrame(records_overall, columns=[
    "Dataset",
    "Discard %",
    "HvD Mean ± Std"
])

latex_table_overall = df_overall.to_latex(
    index=False,
    escape=False,
    column_format="llccc",
    caption="Hypervolume Difference (HvD) across seeds for each dataset and discard percentage.",
    label="tab:hvd_overall"
)

with open("hv_results_overall.tex", "w") as f:
    f.write(latex_table_overall)
print("Saved overall HV table to hv_results_overall.tex")
