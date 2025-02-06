"""
MAE Results Aggregation and Summary Module
==========================================

This module processes and aggregates Mean Absolute Error (MAE) metrics from multiple experimental 
results files to compute disaggregated statistics such as mean and standard deviation. 
The summarized results are saved as a CSV file.
"""

import pandas as pd
import numpy as np
import json
import os

from src.utils.secondary_utils import load_json_results

def create_summary_table_MAE(metrics_dicts_list, output_file="summary_MAE_test.csv"):
    """
    Creates a summary table from multiple MAE metrics dictionaries and saves it as a CSV file.
    Computes mean and standard deviation for each dataset, category, and key.

    Args:
        metrics_dicts_list (list of dict): List of MAE metrics dictionaries from multiple results files.
        output_file (str): Path to save the output CSV file.

    Returns:
        pd.DataFrame: The summary DataFrame containing mean and standard deviation.
    """
    aggregated_data = {}

    # Iterate through multiple MAE results dictionaries
    for metrics_dict in metrics_dicts_list:
        for dataset, categories in metrics_dict.items():
            for category, category_data in categories.items():
                if category == "overall":
                    continue  

                for sub_key, sub_data in category_data.items():
                    key_identifier = (dataset, category, sub_key)  # Unique key for aggregation

                    if key_identifier not in aggregated_data:
                        aggregated_data[key_identifier] = {
                            "MAE VAL_ACC": [],
                            "MAE VAL_ACC LAST": [],
                            "MAE ENERGY": [],
                            "MAE ENERGY LAST": []
                        }

                    # Collect values from all metric dictionaries
                    aggregated_data[key_identifier]["MAE VAL_ACC"].append(sub_data.get("MAE VAL_ACC", np.nan))
                    aggregated_data[key_identifier]["MAE VAL_ACC LAST"].append(sub_data.get("MAE VAL_ACC LAST", np.nan))
                    aggregated_data[key_identifier]["MAE ENERGY"].append(sub_data.get("MAE ENERGY", np.nan))
                    aggregated_data[key_identifier]["MAE ENERGY LAST"].append(sub_data.get("MAE ENERGY LAST", np.nan))

    # Create a structured list for DataFrame conversion
    summary_data = []
    for (dataset, category, sub_key), values in aggregated_data.items():
        summary_data.append({
            "Dataset": dataset,
            "Category": category,
            "Key": sub_key,
            "MAE VAL_ACC_mean": np.nanmean(values["MAE VAL_ACC"]),
            "MAE VAL_ACC_std": np.nanstd(values["MAE VAL_ACC"]),
            "MAE VAL_ACC LAST_mean": np.nanmean(values["MAE VAL_ACC LAST"]),
            "MAE VAL_ACC LAST_std": np.nanstd(values["MAE VAL_ACC LAST"]),
            "MAE ENERGY_mean": np.nanmean(values["MAE ENERGY"]),
            "MAE ENERGY_std": np.nanstd(values["MAE ENERGY"]),
            "MAE ENERGY LAST_mean": np.nanmean(values["MAE ENERGY LAST"]),
            "MAE ENERGY LAST_std": np.nanstd(values["MAE ENERGY LAST"]),
        })

    # Convert the structured data into a DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Apply data_perc transformation for cifar10
    mask_cifar10 = (summary_df["Dataset"] == "cifar10")
    original_keys = summary_df.loc[mask_cifar10, "Key"]

    transformed_keys = np.where(
        original_keys == "100", "0",
        np.where(
            original_keys == "70", "30",
            np.where(
                original_keys == "30", "70",
                original_keys
            )
        )
    )
    summary_df.loc[mask_cifar10, "Key"] = transformed_keys

    # Save the DataFrame to a CSV file
    summary_df.to_csv(output_file, index=False)
    print(f"Summary MAE saved to CSV: {output_file}")

    return summary_df


if __name__ == "__main__":
    
    folder_path = "src/results_csv"
    mae_results = load_json_results(folder_path) 
    summary_MAE_df = create_summary_table_MAE(mae_results, output_file="src/results_csv/summary_MAE_test1.csv")

    
        
