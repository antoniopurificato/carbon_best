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
import yaml

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

def generate_selected_ranked_results(pareto_results: dict, maximize=[True, False]) -> dict:
    """
    Generate ranked Pareto solutions for two selected weight combinations:
    - Full accuracy emphasis (1.0, 0.0)
    - Equal weighting (0.5, 0.5)

    Args:
        pareto_results (dict): 
            A nested dictionary containing the Pareto solutions.
        maximize (list): 
            A list indicating whether each objective should be maximized (True) or minimized (False).

    Returns:
        dict: 
            A dictionary with ranked results for the selected weights, using keys like:
            - "weights_1.0_0.0"
            - "weights_0.5_0.5"
    """
    selected_weights = [[1.0, 0.0], [0.5, 0.5]]
    results_dict = {}

    for weights in selected_weights:
        ranker = RankingSolutions(pareto_results, weights, maximize=maximize)
        ranked_results = ranker.rank_all()

        key = f"weights_{weights[0]:.1f}_{weights[1]:.1f}"
        results_dict[key] = ranked_results

    return results_dict



if __name__ == "__main__":

    with open("NAS/configs/predictor_config_NAS.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    seed = config["seed"]

    pareto_results = load_json_results("NAS/results_NAS", f"pareto_results_NAS_{seed}") #substitute with results_NAS

    sova_dict = {}

    for idx, res in enumerate(pareto_results, start=1):
        # Generate the ranked results for this particular run
        ranked_results_dict = generate_selected_ranked_results(res)
        # Max and min for denormalization Accuracy
        min_val = 0.0060013337060809135
        max_val = 0.9973541498184204
        normalized_target = 0.95279187

        target_raw_accuracy = normalized_target * (max_val - min_val) + min_val

        # Extract top-1 predicted entries
        best_05_05 = ranked_results_dict["weights_0.5_0.5"]["cifar10"]["100"]["predicted"].head(1).to_dict(orient="records")[0]
        best_1_0 = ranked_results_dict["weights_1.0_0.0"]["cifar10"]["100"]["predicted"].head(1).to_dict(orient="records")[0]

        # Replace true_ACC with denormalized version to find matching in nasbench101
        denormalized_05_05 =  best_05_05["true_ACC"] * (max_val - min_val) + min_val
        denormalized_1_0 =best_1_0["true_ACC"] * (max_val - min_val) + min_val

        best_05_05["denorm_ACC"] = denormalized_05_05
        best_1_0["denorm_ACC"] = denormalized_1_0

        # Structure results in desired format
        final_results = {
            "GREEN-NAS": {
                "weights_0.5_0.5": best_05_05,
                "weights_1.0_0.0": best_1_0
            }
        }

        # Save to JSON file
        with open(f"NAS/results_NAS/best_GREEN_NAS_{seed}.json", "w") as f: #substitute with results_NAS
            json.dump(final_results, f, indent=4)
        print(f'Best configurations saved.')