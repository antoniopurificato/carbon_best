import json
from collections import defaultdict

import numpy as np
import pandas as pd

def create_summary_table_from_diff(diff_dict, output_file="summary_MAE_diff.csv"):
    """
    Creates a summary table from a difference dictionary of MAE values and saves it as a CSV.
    This assumes the dictionary contains raw delta values (not a list of metrics from multiple runs).

    Args:
        diff_dict (dict): A nested dictionary of differences in MAE metrics.
        output_file (str): Path to save the output CSV file.

    Returns:
        pd.DataFrame: The summary DataFrame containing the raw deltas.
    """
    summary_data = []

    for dataset, categories in diff_dict.items():
        for category, category_data in categories.items():
            if category == "overall":
                continue  

            for sub_key, sub_data in category_data.items():
                summary_data.append({
                    "Dataset": dataset,
                    "Category": category,
                    "Key": sub_key,
                    "Δ MAE VAL_ACC": sub_data.get("MAE VAL_ACC", np.nan),
                    "Δ MAE VAL_ACC LAST": sub_data.get("MAE VAL_ACC LAST", np.nan),
                    "Δ MAE ENERGY": sub_data.get("MAE ENERGY", np.nan),
                    "Δ MAE ENERGY LAST": sub_data.get("MAE ENERGY LAST", np.nan),
                })

    summary_df = pd.DataFrame(summary_data)
    return summary_df



def recursive_diff(dict1, dict2):
    diff = {}
    for key in dict1:
        if key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                nested_diff = recursive_diff(dict1[key], dict2[key])
                if nested_diff:
                    diff[key] = nested_diff
            elif isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
                diff[key] = abs(dict1[key] - dict2[key])
            else:
                continue  # Ignore non-numeric or mismatched entries
    return diff


with open("src/results_csv/results_MAE_test_reb.json", "r") as f:
    dict_a = json.load(f)
with open("src/results_csv/results_MAE_test.json", "r") as f:
    dict_b = json.load(f)

# Compute the differences
diff_dict = recursive_diff(dict_a, dict_b)
print(diff_dict)

summary_df = create_summary_table_from_diff(diff_dict, "summary_MAE_test_single.csv")

# Transform cifar10's data_perc keys
mask_cifar10 = (summary_df["Dataset"] == "cifar10") & (summary_df["Category"] == "data_perc")
original_keys = summary_df.loc[mask_cifar10, "Key"]
transformed_keys = original_keys.replace({"100": "0", "70": "30", "30": "70"})
summary_df.loc[mask_cifar10, "Key"] = transformed_keys

# Save to CSV
print(summary_df)
output_file="difference_MAE_rebuttal.csv"
summary_df.to_csv(output_file, index=False)
print(f"Summary of MAE differences saved to: {output_file}")

