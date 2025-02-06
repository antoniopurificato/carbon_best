"""
Pareto Metrics Extraction and Aggregation Module
================================================

This module extracts, aggregates, and summarizes metrics from multiple Pareto results 
obtained during multi-objective optimization experiments. It is designed to compare the 
performance of different experimental configurations and provide statistical summaries, 
including mean and standard deviation of key evaluation metrics.
"""
import json
import pandas as pd
import numpy as np
import warnings

from src.utils.secondary_utils import load_json_results

def extract_metrics_from_multiple_results(pareto_results_list, target_data_perc=100, output_csv="mean_metrics_pareto_prova1.csv"):
    """
    Extracts and compares metrics for each dataset from multiple Pareto results.
    Computes mean and standard deviation for `data_perc == target_data_perc`;
    for "foursquare_tky" and "rotten_tomatoes" uses `data_perc=0` instead of `100`.

    Args:
        pareto_results_list (list of dict): List of Pareto results dictionaries.
        target_data_perc (int, optional): The data percentage to filter (default is 100).
        output_csv (str, optional): Filename for saving the extracted metrics (default is "mean_metrics_pareto.csv").

    Returns:
        pd.DataFrame: A DataFrame summarizing the mean and standard deviation for each dataset.
    """
    results_list = []

    # Get ALL dataset keys 
    dataset_keys = set.union(*(set(pareto.keys()) for pareto in pareto_results_list))

    for dataset in dataset_keys:
        # Adjust `data_perc` for "foursquare_tky" and "rotten_tomatoes"
        if dataset == "foursquare_tky" or dataset == "rotten_tomatoes":
            adjusted_data_perc = 0  # Take data_perc=0 but store as 100
            stored_data_perc = 100
        else:
            adjusted_data_perc = target_data_perc
            stored_data_perc = target_data_perc

        adjusted_data_perc_str = str(adjusted_data_perc)  
        metrics_list = []  # List to store metrics from each Pareto result

        for i, pareto_results in enumerate(pareto_results_list):
            # Check if dataset exists in this Pareto result
            if dataset not in pareto_results:
                warnings.warn(f"WARNING: {dataset} missing from Pareto Results {i+1}")
                continue

            # Convert data_perc keys to int where possible
            available_keys = {str(k): v for k, v in pareto_results[dataset].items()}
            if adjusted_data_perc_str in available_keys:
                correct_key = adjusted_data_perc_str
            elif adjusted_data_perc in pareto_results[dataset]:
                correct_key = adjusted_data_perc
            else:
                warnings.warn(f"WARNING: data_perc={adjusted_data_perc} not found in {dataset} for Pareto Results {i+1}")
                continue

            # Extract metrics
            metrics = pareto_results[dataset][correct_key]["metrics"]
            metrics_flat = {}

            for metric_name, value in metrics.items():
                if isinstance(value, dict):  
                    if "value" in value:
                        metrics_flat[metric_name] = value["value"]
                    else:
                        for sub_key, sub_value in value.items():
                            metrics_flat[f"{metric_name}_{sub_key}"] = sub_value
                else:
                    metrics_flat[metric_name] = value  # Direct numerical value case

            metrics_list.append(metrics_flat)

        if metrics_list:

            # Convert list of dictionaries to DataFrame
            metrics_df = pd.DataFrame(metrics_list)

            # Compute mean and standard deviation
            mean_metrics = metrics_df.mean().to_dict()
            std_metrics = metrics_df.std().to_dict()

            # Store results in structured format
            row = {"Dataset": dataset, "data_perc": stored_data_perc}
            row.update({f"{key}_mean": mean for key, mean in mean_metrics.items()})
            row.update({f"{key}_std": std for key, std in std_metrics.items()})
            results_list.append(row)
        else:
            warnings.warn(f"WARNING: No valid metrics found for {dataset} at data_perc={adjusted_data_perc}")

    # Convert to DataFrame for visualization
    results_df = pd.DataFrame(results_list)

    if results_df.empty:
        raise ValueError("ERROR: Final results DataFrame is empty!")
    else:
        # Save the DataFrame to CSV
        results_df.to_csv(output_csv, index=False)
        print(f" Metrics saved to CSV: {output_csv}")

    return results_df



if __name__ == "__main__":
    pareto_results_list = load_json_results("src/results_csv", "pareto_results")
    metrics_comparison_df = extract_metrics_from_multiple_results(pareto_results_list, target_data_perc=100, output_csv="src/results_csv/mean_metrics_pareto_test.csv")