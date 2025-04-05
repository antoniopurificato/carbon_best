import json
from collections import defaultdict
import os
import numpy as np
import pandas as pd


def create_summary_table_from_diff(diff_dict):
    """
    Converts a nested dictionary of absolute differences in MAE metrics into a flat summary DataFrame.

    The input dictionary should be structured as:
    {
        dataset_name: {
            category_name: {
                key: {
                    "MAE VAL_ACC": float,
                    "MAE VAL_ACC LAST": float,
                    "MAE ENERGY": float,
                    "MAE ENERGY LAST": float
                }
            }
        }
    }

    Args:
        diff_dict (dict): A nested dictionary containing MAE delta values per dataset/category/key.

    Returns:
        pd.DataFrame: A DataFrame with columns for dataset, category, key, and the respective delta values.
    """
    summary_data = []

    for dataset, categories in diff_dict.items():
        for category, category_data in categories.items():
            if category == "overall":
                continue  # Skip the "overall" category if present

            for sub_key, sub_data in category_data.items():
                summary_data.append(
                    {
                        "Dataset": dataset,
                        "Category": category,
                        "Key": sub_key,
                        "Δ MAE VAL_ACC": sub_data.get("MAE VAL_ACC", np.nan),
                        "Δ MAE VAL_ACC LAST": sub_data.get("MAE VAL_ACC LAST", np.nan),
                        "Δ MAE ENERGY": sub_data.get("MAE ENERGY", np.nan),
                        "Δ MAE ENERGY LAST": sub_data.get("MAE ENERGY LAST", np.nan),
                    }
                )

    return pd.DataFrame(summary_data)


def recursive_diff(dict1, dict2):
    """
    Recursively computes the absolute differences between two nested dictionaries.

    Only numeric differences (float or int) are calculated; other types are ignored.

    Args:
        dict1 (dict): First input dictionary.
        dict2 (dict): Second input dictionary with the same structure.

    Returns:
        dict: A nested dictionary containing absolute differences for matching numeric keys.
    """
    diff = {}
    for key in dict1:
        if key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                nested_diff = recursive_diff(dict1[key], dict2[key])
                if nested_diff:
                    diff[key] = nested_diff
            elif isinstance(dict1[key], (int, float)) and isinstance(
                dict2[key], (int, float)
            ):
                diff[key] = abs(dict1[key] - dict2[key])
            else:
                continue  # Ignore non-numeric or mismatched entries
    return diff


if __name__ == "__main__":
    output_folder = "src/results_csv"

    # Load JSON result files for comparison
    with open(
        os.path.join(output_folder, "results_MAE_test_sanity_check.json"), "r"
    ) as f:
        dict_a = json.load(f)

    with open(os.path.join(output_folder, "results_MAE_test.json"), "r") as f:
        dict_b = json.load(f)

    # Compute the absolute differences between the two result dictionaries
    diff_dict = recursive_diff(dict_a, dict_b)
    print("Difference Dictionary:")
    print(diff_dict)

    # Create summary table from the difference dictionary
    summary_df = create_summary_table_from_diff(diff_dict)

    # Apply key transformation for CIFAR-10 'data_perc' category (flip percentages)
    mask_cifar10 = (summary_df["Dataset"] == "cifar10") & (
        summary_df["Category"] == "data_perc"
    )
    original_keys = summary_df.loc[mask_cifar10, "Key"]
    transformed_keys = original_keys.replace({"100": "0", "70": "30", "30": "70"})
    summary_df.loc[mask_cifar10, "Key"] = transformed_keys

    # Save the summary table to a CSV file
    output_file = "difference_MAE_sanity_check.csv"
    output_path = os.path.join(output_folder, output_file)
    summary_df.to_csv(output_path, index=False)

    print("Summary DataFrame:")
    print(summary_df)
    print(f"Summary of MAE differences saved to: {output_path}")
