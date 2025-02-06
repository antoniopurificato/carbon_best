"""
NDCG and SOVA Metrics Summary and Visualization Module
======================================================

This module processes and summarizes Normalized Discounted Cumulative Gain (NDCG) 
and Set-based Order Value Alignment (SOVA) metrics from multiple experimental runs 
across different datasets and data percentages, leveraging the dictionary created 
in "3_get_pareto_ranking_metrics". 
"""

import numpy as np
import csv
import json 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def pick_data_perc(ds_name, sova_dict):
    """
    Selects the most appropriate `data_perc` key for the given dataset across all runs.

    The selection follows a priority order:
        - '100' if available
        - '0' if '100' is not available
        - Otherwise, selects the first sorted key

    Args:
        ds_name (str): The name of the dataset for which to select the `data_perc`.
        sova_dict (dict): The nested dictionary containing NDCG and SOVA metrics across multiple runs.

    Returns:
        Optional[str]: The selected `data_perc` key as a string, or None if no keys are available.
    """

    all_data_percs_for_ds = set()
    for run_idx, run_data in sova_dict.items():
        ds_dict = run_data.get(ds_name, {})
        all_data_percs_for_ds.update(ds_dict.keys())

    if not all_data_percs_for_ds:
        return None

    if "100" in all_data_percs_for_ds:
        return "100"
    elif "0" in all_data_percs_for_ds:
        return "0"
    else:
        sorted_percs = sorted(all_data_percs_for_ds)
        return sorted_percs[0] if sorted_percs else None
    
########################## NDCG  TABLE ####################################Ã 

def create_ndcg_table(sova_dict, dataset_names=None):
    """
    Aggregates NDCG values across multiple experimental runs and computes their 
    mean and standard deviation for each dataset.

    Args:
        sova_dict (dict): A nested dictionary containing NDCG and SOVA metrics from multiple runs 
                          and configurations.
        dataset_names (list of str, optional): List of dataset names to process. If None, all datasets 
                                               found in the input dictionary will be processed.

    Returns:
        list of tuples: A list of tuples where each tuple contains:
                        - ds_name (str): The name of the dataset.
                        - data_perc (str): The selected data percentage key.
                        - mean_ndcg (float): The mean NDCG value.
                        - std_ndcg (float): The standard deviation of NDCG values.
    """

    # If not provided, detect all dataset names automatically
    if dataset_names is None:
        dataset_names = set()
        for run_idx, run_data in sova_dict.items():
            dataset_names.update(run_data.keys())
        dataset_names = sorted(dataset_names)

    table = []
    for ds_name in dataset_names:
        dp = pick_data_perc(ds_name, sova_dict)
        if dp is None:
            # No data perc for this dataset
            continue

        # Gather all NDCG values across runs & weight configs
        ndcg_vals = []
        for run_idx, run_data in sova_dict.items():
            dp_dict = run_data.get(ds_name, {}).get(dp, {})
            for wc, metrics_dict in dp_dict.items():
                ndcg_val = metrics_dict.get("NDCG", None)
                if ndcg_val is not None:
                    ndcg_vals.append(ndcg_val)

        if len(ndcg_vals) == 0:
            continue

        mean_ndcg = np.mean(ndcg_vals)
        std_ndcg  = np.std(ndcg_vals)

        table.append((ds_name, dp, mean_ndcg, std_ndcg))

    return table

##################### SOVA PLOT ##########################

def generate_sova_plot(sova_ndcg_dict, output_file: str = "SOVA_plot.pdf"):
    """
    Generates and saves a plot of SOVA@k metrics across datasets and weight configurations.

    Args:
        sova_ndcg_dict (dict): A nested dictionary containing SOVA and NDCG metrics from multiple runs.
        output_file (str, optional): The output file path for the plot. Default is "SOVA_plot.pdf".
    """
    all_dataset_names = {ds_name for run_data in sova_ndcg_dict.values() for ds_name in run_data.keys()}
    plot_dataset_names = sorted(all_dataset_names)

    fig, axes = plt.subplots(1, len(plot_dataset_names), figsize=(5 * len(plot_dataset_names), 5), sharey=False)
    if len(plot_dataset_names) == 1:
        axes = [axes]

    ks = [1, 5, 10]
    colors = ['#1b9e77', '#d95f02', '#7570b3']
    markers = ['o', 'v', 's', 'p', 'P', '*', 'X', '+', 'D', 'x', 'd']
    custom_handles = [Line2D([0], [0], marker=markers[i], color=colors[i], label=f'SOVA@{k}', 
                             markerfacecolor=colors[i], markersize=10, linestyle='None') for i, k in enumerate(ks)]
    custom_labels = [f'SOVA@{k}' for k in ks]


    for i, ds_name in enumerate(plot_dataset_names):
        ax = axes[i]
        data_perc_to_use = pick_data_perc(ds_name, sova_ndcg_dict)

        if data_perc_to_use is None:
            ax.set_title(f"{ds_name}\n(no data perc found)")
            continue

        all_weight_configs = sorted({
            wc for run_data in sova_ndcg_dict.values() 
            for wc in run_data.get(ds_name, {}).get(data_perc_to_use, {}).keys()
        })

        x_positions = np.arange(len(all_weight_configs))
        x_labels = [wc.split("_")[1] for wc in all_weight_configs]

        for idx, k_val in enumerate(ks):
            y_means, y_stds = [], []

            for wc in all_weight_configs:
                values_across_runs = []
                for run_data in sova_ndcg_dict.values():
                    sova_value = run_data.get(ds_name, {}).get(data_perc_to_use, {}).get(wc, {}).get("SOVA", {}).get(str(k_val))
                    if sova_value is not None:
                        values_across_runs.append(sova_value)

                if values_across_runs:
                    y_means.append(np.mean(values_across_runs))
                    y_stds.append(np.std(values_across_runs))
                else:
                    y_means.append(np.nan)
                    y_stds.append(0.0)

            ax.errorbar(x_positions, y_means, yerr=y_stds, fmt=markers[idx], capsize=4, 
                        label=f"SOVA@{k_val}", color=colors[idx])

        ax.set_title(f"{ds_name}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylim(bottom=0, top=0.4)
        ax.set_xlabel(r"$\omega_{A}$" if i == len(plot_dataset_names) // 2 else "")
        ax.set_ylabel("SOVA" if i == 0 else "")
        ax.grid(True)

    fig.legend(custom_handles, custom_labels, loc='upper center', bbox_to_anchor=(0.52, 0.88), ncol=1)
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=700, bbox_inches='tight')
    print(f"Plot saved to {output_file}")



if __name__ == "__main__":

    with open("src/results_csv/sova_ndcg_dict_test.json", "r") as json_file:
        sova_ndcg_dict = json.load(json_file)

    summary_rows = create_ndcg_table(sova_ndcg_dict)

    with open("src/results_csv/ndcg_summary_test1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "DataPerc", "Mean_NDCG", "Std_NDCG"])
        for ds_name, dp, mean_ndcg, std_ndcg in summary_rows:
            writer.writerow([ds_name, dp, f"{mean_ndcg:.4f}", f"{std_ndcg:.4f}"])
    
    generate_sova_plot(sova_ndcg_dict, output_file="src/results_csv/SOVA_plot.pdf")
    