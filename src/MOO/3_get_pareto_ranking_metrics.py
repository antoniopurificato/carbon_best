"""
Pareto Ranking and Evaluation Module
====================================

This module provides functionality to rank and evaluate Pareto-optimal solutions 
based on multi-objective optimization problems. It includes methods to score 
Pareto solutions using a weighted ranking mechanism and compute various metrics 
to assess the quality of predicted solutions compared to ground-truth solutions.

Main Components:
----------------
1. **RankingSolutions Class:** 
    - Ranks Pareto-optimal solutions using normalized objectives and weighted scores.
    - Handles datasets with multiple configurations and supports flexible weight assignment.

2. **generate_ranked_results Function:** 
    - Generates ranked results for varying weight combinations, allowing exploration 
      of different trade-offs between objectives.

3. **compute_sova Function:** 
    - Computes the Set-based Order Value Alignment (SOVA@k) distance to evaluate 
      how well the predicted Pareto front aligns with the true Pareto front.

4. **compute_ndcg_already_sorted Function:** 
    - Computes the Normalized Discounted Cumulative Gain (NDCG) to assess ranking 
      quality while accounting for ties.

File Input/Output:
------------------
- The script reads input JSON files containing Pareto front results.
- Ranked solutions and computed metrics (SOVA and NDCG) are saved as JSON output.

"""


import json
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

from src.utils.secondary_utils import load_json_results


#################### RANKING PARETO SOLUTIONS ###################

class RankingSolutions:
    """Ranks solutions in the given Pareto fronts based on weighted scoring.
    Args:
        pareto_data (dict): A nested dictionary containing Pareto front solutions. Typically in the form:
            {
                "<dataset_name>": {
                    "<configuration_key>": {
                        "ground_truth": [...],  # list of dicts with columns "true_ACC", "true_EN", ...
                        "predicted": [...],     # list of dicts with columns "predicted_ACC", "predicted_EN", ...
                        "metrics": {...}        # Optional non-Pareto data
                    },}}

        weights (list):A list of floats specifying the weight to assign each objective in the scoring function.
        maximize (list):A list of booleans indicating whether each corresponding objective should be maximized (True) or minimized (False). Used for normalization."""
    
    def __init__(self, pareto_data: dict, weights: list, maximize: list):
 
        self.pareto_data = pareto_data
        self.weights = weights
        self.maximize = maximize

        # Define objective labels for each Pareto set
        self.objective_labels = {
            "ground_truth": ["true_ACC", "true_EN"],
            "predicted": ["predicted_ACC", "predicted_EN"]
        }

    def normalize_objectives(self, df: pd.DataFrame, objectives: list) -> pd.DataFrame:
        """
        Normalize the objective columns of a DataFrame.

        For objectives to be maximized, normalization is performed as:
            (x - min(x)) / (max(x) - min(x))
        For objectives to be minimized, normalization is performed as:
            (max(x) - x) / (max(x) - min(x))

        Args:
            df (pd.DataFrame): 
                The DataFrame containing the columns to be normalized.
            objectives (list): 
                The list of column names corresponding to objectives in the DataFrame.

        Returns:
            pd.DataFrame: 
                A new DataFrame with normalized values for each objective"""
        
        normalized = pd.DataFrame()
        for obj, maximize_flag in zip(objectives, self.maximize):
            if maximize_flag:
                normalized[obj] = (df[obj] - df[obj].min()) / (df[obj].max() - df[obj].min())
            else:
                normalized[obj] = (df[obj].max() - df[obj]) / (df[obj].max() - df[obj].min())
        
        return normalized

    def rank_by_weighted_scoring(self, df: pd.DataFrame, objectives: list) -> pd.DataFrame:
        """
        Rank Pareto solutions using weighted scoring.
        The objectives are normalized and then multiplied by the corresponding weights. 
        The sum of weighted objective scores is used to assign a ranking.

        Args:
            df (pd.DataFrame): 
                The DataFrame containing the Pareto solutions to rank.
            objectives (list): 
                Column names in the DataFrame corresponding to the objectives to score.

        Returns:
            pd.DataFrame: 
                A new DataFrame with an additional "Score" column representing the weighted score 
                of each solution, sorted in descending order of score.
        """
        normalized = self.normalize_objectives(df, objectives)
        scores = normalized.mul(self.weights, axis=1).sum(axis=1)
        return df.assign(Score=scores).sort_values("Score", ascending=False)

    def rank_all(self):
        results = {}

        for dataset, configurations in self.pareto_data.items():
            results[dataset] = {}

            for config_key, config_data in configurations.items():
                results[dataset][config_key] = {}

                for name, solutions in config_data.items():
                    # Skip non-Pareto keys like 'metrics'
                    if name == "metrics":
                        continue

                    objectives = self.objective_labels[name]  
                    df = pd.DataFrame(solutions)
                    ranked = self.rank_by_weighted_scoring(df, objectives)
                    results[dataset][config_key][name] = ranked

        return results


def generate_ranked_results(pareto_results: dict, maximize=[True, False]) -> dict:
    """
    Generate a dictionary of ranked Pareto solutions for varying weight combinations.

    The function generates rankings for different combinations of weights assigned to 
    accuracy and energy (or any two objectives) within the Pareto fronts. 
    The weights range from 0 to 1 (in steps of 0.1) and satisfy the condition 
    w_acc + w_energy = 1.

    Args:
        pareto_results (dict): 
            A nested dictionary containing the Pareto solutions to be ranked. 
            The structure should follow the format expected by the `RankingSolutions` class.
        maximize (list, optional): 
            A list of booleans indicating whether each objective should be maximized (True) 
            or minimized (False). Defaults to [True, False].

    Returns:
        dict: 
            A dictionary containing ranked results for each weight combination. 
            The keys of the dictionary follow the format "weights_<w_acc>_<w_energy>", where 
            <w_acc> and <w_energy> correspond to the accuracy and energy weights, respectively. 
            The values are the ranked results for the respective weight combination.
    
    """
    weights_range = np.arange(0, 1.1, 0.1)  # Generate weights from 0 to 1, step 0.1
    results_dict = {}

    # Generate weight pairs (w_acc, w_energy) such that w_acc + w_energy = 1
    for w_acc in weights_range:
        w_energy = 1 - w_acc
        weights = [w_acc, w_energy]

        # Rank solutions for this weight combination
        ranker = RankingSolutions(pareto_results, weights, maximize=maximize)
        ranked_results = ranker.rank_all()

        # Store the ranked results in the dictionary
        results_dict[f"weights_{w_acc:.1f}_{w_energy:.1f}"] = ranked_results

    return results_dict

###################### COMPUTE METRICS ###########################

##### SOVA 

def compute_sova(true_values_set1: np.ndarray, true_values_set2: np.ndarray, k: int = None, 
                             weights: np.ndarray = None, lambda_decay: float = 1.0) -> float:
    """
    Compute the Set-based Order Value Alignment (SOVA@k) between two sets of true values, 
    handling cases with identical predicted scores and applying rank-based weighting.

    This distance measures the difference between sets of ranked solutions while 
    accounting for objective-specific importance and rank-based decay. The distance 
    is normalized to fall between [0, 1], where lower values indicate better agreement.

    Args:
        true_values_set1 (np.ndarray): 
            Array of shape (n_points, n_objectives + 1) representing the true values of the 
            first set (e.g., true Pareto front). The last column is treated as a non-objective 
            value (e.g., score).
        true_values_set2 (np.ndarray): 
            Array of shape (n_points, n_objectives + 1) representing the true values of the 
            second set (e.g., predicted Pareto front). The last column contains the predicted 
            scores used for ranking.
        k (int, optional): 
            Number of top-ranked unique scores to evaluate (cutoff). If None, evaluates the 
            full set.
        weights (np.ndarray, optional): 
            Weights for each objective, of shape (n_objectives,). If None, all objectives are 
            equally weighted.
        lambda_decay (float, optional): 
            Controls the rate of decay for rank-based weights. Default is 1.0.

    Returns:
        float: 
            The normalized set-based order value distance between the two sets, where a value 
            of 0 indicates perfect alignment and higher values indicate larger discrepancies.
    """
    
    if k is not None:
        # Extract and sort unique scores
        set2_scores = true_values_set2[:, -1]
        unique_scores, unique_indices = np.unique(set2_scores, return_index=True)
        unique_indices_sorted = np.sort(unique_indices)
        
        # Determine the cutoff based on the k-th unique score
        cutoff_idx = unique_indices_sorted[min(k, len(unique_scores)) - 1] + 1
        
        # Slice both sets up to the determined cutoff
        true_values_set1 = true_values_set1[:cutoff_idx]
        true_values_set2 = true_values_set2[:cutoff_idx]

    n1, cols1 = true_values_set1.shape
    n2, cols2 = true_values_set2.shape
    num_objectives = cols1 - 1  # Exclude the last column (score) from objectives

    # Handle weights
    if weights is None:
        weights = np.ones(num_objectives, dtype=float)
    objective_weights = weights / np.sum(weights)  # Normalize weights

    # Group by unique scores and compute rank-based weights
    set2_scores = true_values_set2[:, -1]
    unique_scores, group_idx = np.unique(set2_scores, return_inverse=True)
    rank_weights = np.exp(-lambda_decay * np.arange(len(unique_scores)))
    rank_weights /= rank_weights.sum()  # Normalize to ensure weights sum to 1

    # Compute the total distance
    total_distance = 0.0

    for i, score_val in enumerate(unique_scores):
        # Indices of predicted solutions belonging to the current score group
        pred_indices = np.where(group_idx == i)[0]

        # If no corresponding row in true_values_set1, skip (important for edge cases)
        if i >= n1:
            break

        # Ground-truth row to compare with the predicted group
        ground_truth_row = true_values_set1[i, :num_objectives]

        # Calculate differences for the group
        group_diffs = np.abs(ground_truth_row - true_values_set2[pred_indices, :num_objectives])
        weighted_group_diffs = np.dot(group_diffs, objective_weights)
        avg_group_diff = np.mean(weighted_group_diffs) if len(weighted_group_diffs) > 0 else 0.0

        # Accumulate the distance, weighted by rank
        total_distance += rank_weights[i] * avg_group_diff

    return total_distance  # Sum of rank_weights is already normalized to 1

##### NDCG

def compute_ndcg_already_sorted(df: pd.DataFrame, acc_weight: float, en_weight: float) -> float:
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) with support for ties.

    Args:
        df (pd.DataFrame): 
            The DataFrame containing solutions, assumed to be sorted in predicted ranking order.
        acc_weight (float): 
            The weight assigned to accuracy in the true relevance calculation.
        en_weight (float): 
            The weight assigned to energy (or cost) in the true relevance calculation.

    Returns:
        float: The computed NDCG score, a value between 0 and 1.
    """
     
    # Calculate the "true" relevance for each row
    df["true_relevance"] = (
        acc_weight * df["true_ACC"] +
        en_weight * (1.0 - df["true_EN"])
    )
    
    # Compute the ranks for handling ties (average rank for tied predicted scores)
    df['predicted_rank'] = df['Score'].rank(method='average', ascending=False)

    # Compute DCG, accounting for ties using predicted ranks
    dcg = 0.0
    for idx, row in df.iterrows():
        rank = row["predicted_rank"]
        rel = row["true_relevance"]
        dcg += rel / np.log2(rank + 1)  # +1 to align with log2(rank)

    # Compute the ideal DCG (IDCG) by sorting true relevance scores in descending order
    ideal_scores = np.sort(df["true_relevance"].values)[::-1]
    idcg = 0.0
    for idx, rel in enumerate(ideal_scores):
        idcg += rel / np.log2(idx + 2)  # +2 because idx is 0-based

    return dcg / idcg if idcg > 0 else 0.0



if __name__ == "__main__":

    pareto_results = load_json_results("src/results_csv", "pareto_results")

    sova_dict = {}

    for idx, res in enumerate(pareto_results, start=1):
        # Generate the ranked results for this particular run
        ranked_results_dict = generate_ranked_results(res)

        # Create a sub-dict for this run
        rankig_metrics_dict = {}
        
        # Iterate through the ranked_results_dict
        for weights_config, datasets in ranked_results_dict.items():
            for dataset_name, data_percs in datasets.items():
                for data_perc, results in data_percs.items():
                    # Extract ground_truth and predicted DataFrames
                    ground_truth = results['ground_truth']
                    predicted = results['predicted']
                    
                    # Parse the weight config string, e.g., "weights_0.7_0.3" -> [0.7, 0.3]
                    parts = weights_config.split("_")[1:]
                    parsed_weights = [float(p) for p in parts]

                    # Extract columns and convert to NumPy arrays
                    true_values_set1 = ground_truth[['true_ACC', 'true_EN', 'Score']].to_numpy()
                    true_values_set2 = predicted[['true_ACC', 'true_EN', 'Score']].to_numpy()

                    # Compute SOVD@k for k = [1, 5, 10]
                    sovd_results = {}
                    for k_val in [1, 5, 10]:
                        distance = compute_sova(
                            true_values_set1,
                            true_values_set2,
                            k=k_val,
                            weights=parsed_weights,
                            lambda_decay=1.0
                        )
                        sovd_results[k_val] = distance

                    ndcg_value = compute_ndcg_already_sorted(
                        predicted.copy(),  # pass a copy to avoid altering predicted
                        parsed_weights[0],
                        parsed_weights[1]
                    )

                    # Store in run_sova_dict
                    if dataset_name not in rankig_metrics_dict:
                        rankig_metrics_dict[dataset_name] = {}
                    if data_perc not in rankig_metrics_dict[dataset_name]:
                        rankig_metrics_dict[dataset_name][data_perc] = {}

                    rankig_metrics_dict[dataset_name][data_perc][weights_config] = {
                        "SOVA": sovd_results,
                        "NDCG": ndcg_value
                    }

        
        sova_dict[idx] = rankig_metrics_dict
    output_folder = "src/results_csv/sova_ndcg_dict_test.json"
    with open(output_folder, "w") as f:
        json.dump(sova_dict, f, indent=4)
    print(f'Results saved to {output_folder}!')
