"""
Pareto Front Computation and Evaluation Module
==============================================

This module provides functionality for computing Pareto-optimal solutions from 
datasets containing true and predicted objective metrics, and evaluating the 
quality of predicted solutions compared to ground truth using various metrics.

Key Components:
---------------
1. **ParetoSolutions Class:** 
    - Identifies Pareto-optimal solutions for both ground truth and predicted objective metrics.
    - Filters solutions based on a minimum accuracy threshold and ensures that predicted solutions 
      do not dominate ground truth solutions, maintaining consistency.

2. **EvaluateParetoSolutions Class:** 
    - Evaluates the alignment between predicted and ground truth Pareto fronts using both 
      standard metrics and geometric measures.

3. **Auxiliary Functions:** 
    - Functions for processing datasets, handling serialization, and saving results 
      to JSON files for further analysis.

Main Features:
--------------
- **Pareto Front Computation:** 
    - Finds and filters Pareto-optimal solutions based on specified objectives and thresholds.
    - Ensures that no predicted Pareto solution dominates any ground truth solution.

- **Evaluation Metrics:** 
    - Standard metrics: Precision, Recall, F1-score based on matching true and predicted Pareto points.
    - Geometric distance: Hausdorff distance to quantify the difference between the two Pareto fronts.

File Input/Output:
------------------
- Reads input data from CSV files containing experimental results.
- Computes Pareto fronts and evaluation metrics, saving them to JSON files for further use.
"""


import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
from scipy.spatial.distance import cdist
import json
from datetime import date
import os
import glob
import yaml

class ParetoSolutions:
    """Computes Pareto fronts based on true and predicted objectives and ensures consistency 
    by checking that no predicted Pareto point dominates any ground truth point.

    Args:
        data (pd.DataFrame): 
            The dataset containing objective values for all solutions.
        true_objectives (list): 
            Column names corresponding to the true objective metrics.
        predicted_objectives (list): 
            Column names corresponding to the predicted objective metrics.
        min_acc (float): 
            Minimum accuracy threshold to filter Pareto-optimal solutions.
        maximize (list, optional): 
            A list of booleans indicating whether each objective should be maximized (True) 
            or minimized (False). Default is [True, False]."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        true_objectives: list,
        predicted_objectives: list,
        min_acc: float,
        maximize: list = [True, False]
    ):
        self.data = data
        self.true_objectives = true_objectives
        self.predicted_objectives = predicted_objectives
        self.min_acc = min_acc
        self.maximize= maximize

        # Compute Pareto fronts without filtering
        self.ground_truth_pareto = self._find_pareto_front(self.true_objectives)
        self.predicted_pareto = self._find_pareto_front(self.predicted_objectives)

        # Apply filtering for minimum accuracy thresholds
        self.ground_truth_pareto = self._apply_min_acc_filter(
            self.ground_truth_pareto, self.true_objectives[0]
        )
        self.predicted_pareto = self._apply_min_acc_filter(
            self.predicted_pareto, self.predicted_objectives[0]
        )

        # Ensure no predicted Pareto point dominates ground truth Pareto points
        self._validate_predicted_pareto()


    def _is_pareto_optimal(self, objectives: pd.DataFrame, tol: float = 1e-6) -> np.ndarray:
        """
        Determine which rows in the DataFrame represent Pareto-optimal solutions.

        Args:
            objectives (pd.DataFrame): 
                DataFrame containing the objectives to evaluate.
            tol (float, optional): 
                Tolerance level for comparing objective values. Default is 1e-8.

        Returns:
            np.ndarray: 
                A boolean array where True indicates a Pareto-optimal point.
        """
        objectives = objectives.values  # Convert to NumPy array for performance
        num_points = objectives.shape[0]
        is_optimal = np.ones(num_points, dtype=bool)  # Start with all points as optimal

        for i in range(num_points):
            if is_optimal[i]:
                point = objectives[i]
                # Iteratively check dominance for all other points
                for j in range(num_points):
                    if i != j:
                        other = objectives[j]
                        # Check dominance based on maximization/minimization
                        dominates = all(
                            (other[idx] >= point[idx] - tol if maximize else other[idx] <= point[idx] + tol)
                            for idx, maximize in enumerate(self.maximize)
                        ) and any(
                            (other[idx] > point[idx] + tol if maximize else other[idx] < point[idx] - tol)
                            for idx, maximize in enumerate(self.maximize)
                        )
                        if dominates:
                            is_optimal[i] = False
                            break  # No need to check further if already dominated

        return is_optimal


    def _find_pareto_front(self, objectives):
        """
        Identify the Pareto-optimal solutions for the given objectives.

        Args:
            objectives (list): 
                Column names of the objectives to evaluate.

        Returns:
            pd.DataFrame: 
                DataFrame containing Pareto-optimal solutions for the specified objectives.
        """
        is_pareto = self._is_pareto_optimal(
            self.data[objectives]
        )
        return self.data.loc[is_pareto].copy()

    def _apply_min_acc_filter(self, pareto_front, acc_column):
        """
        Filter Pareto front based on the minimum accuracy threshold.

        Args:
            pareto_front (pd.DataFrame): 
                DataFrame containing the initial Pareto-optimal solutions.
            acc_column (str): 
                Column name corresponding to the accuracy metric.

        Returns:
            pd.DataFrame: 
                Filtered Pareto front with solutions meeting the minimum accuracy threshold.
        """

        return pareto_front[pareto_front[acc_column] >= self.min_acc]

    def _validate_predicted_pareto(self):
        """
        Ensure that no point in the predicted Pareto front dominates any point 
        in the ground truth Pareto front.
        
        Raises:
            ValueError: If any predicted Pareto point dominates a ground truth point.
        """
        for _, pred_point in self.predicted_pareto.iterrows():
            for _, gt_point in self.ground_truth_pareto.iterrows():
                # Check dominance based on true values
                pred_dominates = all(
                    (pred_point[obj] >= gt_point[obj] if maximize else pred_point[obj] <= gt_point[obj])
                    for obj, maximize in zip(self.true_objectives, self.maximize)
                ) and any(
                    (pred_point[obj] > gt_point[obj] if maximize else pred_point[obj] < gt_point[obj])
                    for obj, maximize in zip(self.true_objectives, self.maximize)
                )

                if pred_dominates:
                    raise ValueError(
                        f"Predicted Pareto point {pred_point} dominates ground truth Pareto point {gt_point} based on true values. This violates consistency."
                    )
                
    def get_pareto_dfs (self):
        """
        Retrieve the filtered Pareto-optimal DataFrames for true and predicted objectives.

        Returns:
            tuple: 
                A tuple containing two DataFrames:
                - ground_truth_pareto (pd.DataFrame): Filtered Pareto front for true objectives.
                - predicted_pareto (pd.DataFrame): Filtered Pareto front for predicted objectives.
        """
        return self.ground_truth_pareto, self.predicted_pareto


class EvaluateParetoSolutions:
    """
    Evaluate and compare Pareto-optimal solutions using various metrics, including 
    standard metrics (precision, recall, F1-score) and geometric distance measures 
    (Hausdorff distance).

    Args:
        ground_truth_pareto (pd.DataFrame): 
            DataFrame containing the ground truth Pareto front with columns such as 
            "true_ACC" and "true_EN".
        predicted_pareto (pd.DataFrame): 
            DataFrame containing the predicted Pareto front with columns such as 
            "predicted_ACC" and "predicted_EN".
    """
    def __init__(self, ground_truth_pareto, predicted_pareto):
        self.ground_truth_pareto = ground_truth_pareto
        self.predicted_pareto = predicted_pareto
        self.true_pareto_set = self.ground_truth_pareto[['true_ACC', 'true_EN']].to_numpy()
        self.predicted_pareto_set = self.predicted_pareto[['predicted_ACC', 'predicted_EN']].to_numpy()

    def compute_metrics(self, tp, fp, fn):
        """
        Compute standard evaluation metrics: recall and F1-score.

        Args:
            tp (int): Number of true positives.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.

        Returns:
            dict: A dictionary containing the computed metrics:
                - recall: Recall score.
                - f1_score: F1 score.
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"recall": recall, "f1_score": f1_score}  # "precision": precision,

    def get_standard_metrics(self):
        """
        Compute precision, recall, and F1-score for strict, loose, and relaxed matching criteria.

        Matching criteria:
            - Strict Matching: Matches based on (key_str + epoch).
            - Loose Matching: Matches based on key_str only, with frequency-based weighting.
            - Relaxed Matching: Matches based on (key_str + epoch within ±5).

        Returns:
            dict: A dictionary containing standard metrics for the following matching levels:
                - with_epoch: Metrics for strict matching.
                - ignore_epoch: Metrics for loose matching with frequency weighting.
                - relaxed_epoch: Metrics for relaxed matching with ±5 epoch tolerance.
        """
        # 1) STRICT MATCHING (key_str + epoch)
        ground_truth_with_epoch = set(
            self.ground_truth_pareto[["key_str", "epoch"]].itertuples(index=False, name=None)
        )
        predicted_with_epoch = set(
            self.predicted_pareto[["key_str", "epoch"]].itertuples(index=False, name=None)
        )
        tp_with_epoch = len(ground_truth_with_epoch & predicted_with_epoch)
        fp_with_epoch = len(predicted_with_epoch - ground_truth_with_epoch)
        fn_with_epoch = len(ground_truth_with_epoch - predicted_with_epoch)

        # 2) LOOSE MATCHING (key_str only)
        ground_truth_ignore_epoch = set(self.ground_truth_pareto["key_str"])
        predicted_ignore_epoch = self.predicted_pareto["key_str"]

        # Use Counter for frequencies
        predicted_counts = Counter(predicted_ignore_epoch)
        total_pred_count = sum(predicted_counts.values())  

        # Weighted true positives, false positives, false negatives
        tp_ignore_epoch_weighted = sum(
            predicted_counts[key] for key in ground_truth_ignore_epoch if key in predicted_counts
        )
        fp_ignore_epoch_weighted = total_pred_count - tp_ignore_epoch_weighted
        predicted_counts_keys = set(predicted_counts.keys())
        fn_ignore_epoch_weighted = len(ground_truth_ignore_epoch - predicted_counts_keys)

        # 3) RELAXED MATCHING (key_str + epoch within ±5)
        predicted_dict = defaultdict(list)
        for _, row in self.predicted_pareto.iterrows():
            predicted_dict[row["key_str"]].append(row["epoch"])

        tp_relaxed_epoch = 0
        matched_keys = set()

        for key, true_epoch in self.ground_truth_pareto.set_index("key_str")["epoch"].items():
            if key in predicted_dict:
                # Check epochs within ±5
                matching_epochs = [
                    pred_epoch for pred_epoch in predicted_dict[key]
                    if abs(pred_epoch - true_epoch) <= 5
                ]
                tp_relaxed_epoch += len(matching_epochs)
                if matching_epochs:
                    matched_keys.add(key)

        # Compute false negatives and false positives
        fn_relaxed_epoch = len(self.ground_truth_pareto) - len(matched_keys)
        predicted_total = sum(len(epochs) for epochs in predicted_dict.values())
        fp_relaxed_epoch = predicted_total - tp_relaxed_epoch

        return {
            "with_epoch": self.compute_metrics(tp_with_epoch, fp_with_epoch, fn_with_epoch),
            "ignore_epoch": self.compute_metrics(
                tp_ignore_epoch_weighted,
                fp_ignore_epoch_weighted,
                fn_ignore_epoch_weighted
            ),
            "relaxed_epoch": self.compute_metrics(
                tp_relaxed_epoch,
                fp_relaxed_epoch,
                fn_relaxed_epoch
            )
        }

    def hausdorff_distance(self):
        """
        Compute the Hausdorff distance between the ground truth and predicted Pareto fronts.

        The Hausdorff distance measures the maximum distance of a point in one set 
        to the nearest point in the other set. It is computed as:
            max{sup_{a in A} inf_{b in B} d(a, b), sup_{b in B} inf_{a in A} d(b, a)}

        Returns:
            float: The computed Hausdorff distance.
        """
        distances = cdist(self.true_pareto_set, self.predicted_pareto_set)  # Pairwise distances
        forward = distances.min(axis=1).max()  # sup_{a in A} inf_{b in B}
        backward = distances.min(axis=0).max()  # sup_{b in B} inf_{a in A}
        return max(forward, backward)

def process_multiple_dfs(
    dfs, test_names, epoch_limits, group_columns=["data_perc", "lr", "model_name"]):
    """
    Process multiple DataFrames, calculate MAEs, and return structured results for all epochs and the last epoch.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames to process.
        test_names (list of str): List of test names corresponding to each DataFrame.
        epoch_limits (dict): Dictionary with test names as keys and epoch limits as values.
        group_columns (list of str): Columns to group by for MAE calculation.

    Returns:
        dict: Nested dictionary with MAE results for each test name, including all-epoch and last-epoch metrics.
    """
    results = {}

    for df, test_name in zip(dfs, test_names):
        # Filter out observations where lr == 0.01
        #df = df[df["lr"] != 0.01]
        # Cap predicted_ACC values greater than 1
        df.loc[df['predicted_ACC'] > 1, 'predicted_ACC'] = 1

        # Floor predicted_EN values less than 0
        df.loc[df['predicted_EN'] < 0, 'predicted_EN'] = 0

        test_results = {}

        # Filter for the last epoch based on the provided limit
        epoch_limit = epoch_limits.get(test_name, None)
        if epoch_limit is not None:
            df = df[df["epoch"] <= epoch_limit]  # Data for all epochs up to the limit

        # Calculate overall MAE (across all epochs)
        overall_mask_first = df["true_ACC"] != -1
        overall_mask_rest = df["true_EN"] != -1

        overall_diff_first = np.abs(df["predicted_ACC"][overall_mask_first] - df["true_ACC"][overall_mask_first])
        overall_diff_rest = np.abs(df["predicted_EN"][overall_mask_rest] - df["true_EN"][overall_mask_rest])

        overall_mae_first = np.mean(overall_diff_first)
        overall_mae_rest = np.mean(overall_diff_rest)


        # Combine overall and last-epoch MAEs under the "overall" key
        test_results["overall"] = {
            "MAE VAL_ACC": overall_mae_first,
            "MAE ENERGY": overall_mae_rest,
        }

        # Calculate MAEs for each group column (for all epochs and last epoch)
        for column in group_columns:
            column_results = {}

            for value in df[column].unique():
                # Filter data for the current value
                subset = df[df[column] == value]

                # MAE for all epochs
                subset_mask_first = subset["true_ACC"] != -1
                subset_mask_rest = subset["true_EN"] != -1

                subset_diff_first = np.abs(
                    subset["predicted_ACC"][subset_mask_first] - subset["true_ACC"][subset_mask_first]
                )
                subset_diff_rest = np.abs(
                    subset["predicted_EN"][subset_mask_rest] - subset["true_EN"][subset_mask_rest]
                )

                subset_mae_first = np.mean(subset_diff_first)
                subset_mae_rest = np.mean(subset_diff_rest)


                # Store results
                column_results[value] = {
                    "MAE VAL_ACC": subset_mae_first,
                    "MAE ENERGY": subset_mae_rest
                }

            # Store results for the column
            test_results[column] = column_results

        # Add test results to the overall dictionary
        results[test_name] = test_results

    return results

def save_results_to_json(results, output_file):
    """
    Save the results dictionary to a JSON file.

    Args:
        results (dict): The results dictionary to save.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {output_file}")


def convert_to_serializable(obj):
    """
    Recursively convert non-serializable objects, such as numpy data types, 
    to standard Python types that can be serialized to JSON.

    Args:
        obj: 
            The object to be converted, which may contain nested structures 
            with numpy arrays, floats, or ints.

    Returns:
        A Python object  that is JSON serializable."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert numpy float to Python float
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert numpy int to Python int
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj  # Return the object as is if it's already JSON serializable
    
def convert_pareto_results_to_serializable(pareto_results):
    """
    Convert a nested dictionary of Pareto results into a JSON-serializable format.

    This function processes the ground truth and predicted Pareto fronts (stored as DataFrames)
    and converts them into lists of dictionaries for JSON serialization. It also includes metrics
    directly since they are already compatible with JSON.

    Args:
        pareto_results (dict): 
            A nested dictionary containing test names, data percentages, and Pareto-related information.
            Structure:
            {
                "test_name": {
                    "data_perc": {
                        "ground_truth": pd.DataFrame(...),
                        "predicted": pd.DataFrame(...),
                        "metrics": {...}
                    }
                }
            }

    Returns:
        dict: 
            A new dictionary with all data converted to JSON-serializable formats, including:
            - "ground_truth": List of dictionaries representing the ground truth Pareto front.
            - "predicted": List of dictionaries representing the predicted Pareto front.
            - "metrics": Metrics directly as-is."""

    serializable_results = {}
    for test_name, data_perc_subsets in pareto_results.items():
        serializable_results[test_name] = {}
        for data_perc, pareto_data in data_perc_subsets.items():
            # Convert ground truth and predicted DataFrames to lists of dicts
            ground_truth_serialized = pareto_data["ground_truth"].to_dict(orient="records")
            predicted_serialized = pareto_data["predicted"].to_dict(orient="records")
            
            # Metrics are already JSON-compatible, so add directly
            metrics_serialized = pareto_data["metrics"]
            
            # Add serialized data to the dictionary
            serializable_results[test_name][data_perc] = {
                "ground_truth": ground_truth_serialized,
                "predicted": predicted_serialized,
                "metrics": metrics_serialized
            }
    return serializable_results


def convert_keys_to_python_int(d):
    """
    Recursively convert keys of a dictionary from numpy integer types to standard Python integers.

    This function is useful when dealing with dictionaries whose keys are numpy int types, 
    which are not directly JSON serializable.

    Args:
        d (dict or list): 
            The input dictionary or list containing keys/values that may be numpy int types.

    Returns:
        dict or list: 
            A new dictionary or list with all keys and values converted to Python-native types."""
    
    if isinstance(d, dict):
        return {int(k) if isinstance(k, (np.int64, np.int32)) else k: convert_keys_to_python_int(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_python_int(i) for i in d]
    else:
        return d


##########
if __name__ == "__main__":
    folder = "NAS/results_NAS_bench101" #chane in results_NAS as in config
    
    with open("NAS/configs/predictor_config_NAS.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    seed = config["seed"]
    pattern = os.path.join(folder, f"cifar10_{seed}_chunck_*.csv")
    print(pattern)
    csv_files = sorted(glob.glob(pattern))


    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched pattern: {pattern}")

    print(f"Found {len(csv_files)} CSV files. Loading...")

    res_dfs = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(res_dfs, ignore_index=True)

    test_names = ["cifar10"]
    res_dfs = [full_df]
    test_names = ["cifar10",]
    epoch_limits = {"cifar10": 108}
    
    #Set accuracy thresholds
    accuracy_thresholds = [0.90]

    # Process multiple test results and save MAE results to JSON
    #results=process_multiple_dfs(dfs=res_dfs, test_names=test_names, epoch_limits=epoch_limits, group_columns=["data_perc", "lr", "model_name"])
    #serializable_results = convert_to_serializable(results)

   # output_file = "src/results_csv/results_MAE_test.json"
    #with open(output_file, "w") as json_file:
      #  json.dump(serializable_results, json_file, indent=4)
    #print("Computed and saved MAE results")

    # Extract Pareto frontiers and metrics and save to JSON
    subsets = {}
    for i, df in enumerate(res_dfs):
        test_name = test_names[i]
        accuracy_threshold = accuracy_thresholds[i] 

        subsets[test_name] = {}  

        # subsets for each unique value in `data_perc`
        for value in df["data_perc"].unique():
            subsets[test_name][value] = {
                "data": df[df["data_perc"] == value],  
                "accuracy_threshold": accuracy_threshold  
            }

    pareto_results= {}  

    for test_name, data_perc_subsets in subsets.items():
        pareto_results[test_name] = {}
        metrics_accumulator = {} 

        for data_perc, subset_info in data_perc_subsets.items():
            data = subset_info["data"]
            accuracy_threshold = subset_info["accuracy_threshold"]

            # Initialize ParetoSolutions optimizer
            optimizer = ParetoSolutions(
                data=data,
                true_objectives=["true_ACC", "true_EN"],
                predicted_objectives=["predicted_ACC", "predicted_EN"],
                min_acc=accuracy_threshold,
                maximize=[True, False]
            )

            # Get Pareto frontiers
            ground_truth_pareto, predicted_pareto = optimizer.get_pareto_dfs()
            print(f"Computed pareto fronts for {test_name}- data_perc{data_perc}. Computing metrics... ")


            # Initialize evaluator with ground truth and predicted Pareto sets
            evaluator = EvaluateParetoSolutions(
                ground_truth_pareto=ground_truth_pareto,
                predicted_pareto=predicted_pareto
            )

            # Compute standard metrics
            metrics = evaluator.get_standard_metrics()

            # Swap the structure of metrics to group by metric type
            swapped_metrics = {}
            for level, level_metrics in metrics.items():
                for metric_name, value in level_metrics.items():
                    if metric_name not in swapped_metrics:
                        swapped_metrics[metric_name] = {}
                    swapped_metrics[metric_name][level] = value

            # Add Pareto sets and metrics to the results dictionary
            pareto_results[test_name][data_perc] = {
                "ground_truth": ground_truth_pareto,  
                "predicted": predicted_pareto,
                "metrics": swapped_metrics  
            }
           
    serializable_pareto_results = convert_keys_to_python_int(convert_pareto_results_to_serializable(pareto_results))

    with open(f"NAS/results_NAS/pareto_results_NAS_{seed}.json", "w") as json_file:
        json.dump(serializable_pareto_results, json_file, indent=4)



