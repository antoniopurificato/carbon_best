import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
from scipy.spatial.distance import cdist
import numpy as np
from pymoo.indicators.hv import HV

from src.utils.plot import *


class ParetoSolutions:
    def __init__(
        self,
        data: pd.DataFrame,
        true_objectives: list,
        predicted_objectives: list,
        min_acc: float,
        maximize: list = [True, False]
    ):
        """
        Multi-objective Pareto solution finder.

        Args:
            data (pd.DataFrame): Dataset containing objectives.
            true_objectives (list): Columns for the true objective values (e.g., ["true_ACC", "true_EM"]).
            predicted_objectives (list): Columns for the predicted objective values.
            min_acc (float): Minimum acceptable accuracy threshold for filtering.
        """
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


    def _is_pareto_optimal(self, objectives: pd.DataFrame, tol: float = 1e-8) -> np.ndarray:
        """
        Identify Pareto-optimal points based on multiple objectives, accounting for maximization or minimization

        Returns:
            np.ndarray: Boolean array indicating Pareto-optimal points.
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
        Compute Pareto front for the provided objectives.
        """
        is_pareto = self._is_pareto_optimal(
            self.data[objectives]
        )
        return self.data.loc[is_pareto].copy()

    def _apply_min_acc_filter(self, pareto_front, acc_column):
        """
        Filter Pareto front based on the minimum accuracy threshold."""

        return pareto_front[pareto_front[acc_column] >= self.min_acc]

    def _validate_predicted_pareto(self):
        """
        Ensure that no point in the predicted Pareto front dominates any point in the ground truth Pareto front
        when evaluated based on true values.
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
        return self.ground_truth_pareto, self.predicted_pareto
class EvaluateParetoSolutions:
    def __init__(self, ground_truth_pareto, predicted_pareto):
        self.ground_truth_pareto = ground_truth_pareto
        self.predicted_pareto = predicted_pareto
        self.true_pareto_set = self.ground_truth_pareto [['true_ACC', 'true_EM']].to_numpy()
        self.predicted_pareto_set = self.predicted_pareto [['predicted_ACC', 'predicted_EM']].to_numpy()

    def compute_metrics(self, tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return {"precision": precision, "recall": recall, "f1_score": f1_score}
    
    def get_standard_metrics(self):
        """
        Compute standard metrics for different matching levels:
        - Strict matching (key_str (==model,bs,lr) + epoch).
        - Loose matching (key_str (==model,bs,lr) only).
        - Relaxed matching (key_str(==model,bs,lr) + epoch within ±5).
        Includes frequency-weighted metrics for loose matching.
        """
        # Strict matching (key_str + epoch)
        ground_truth_with_epoch = set(self.ground_truth_pareto[["key_str", "epoch"]].itertuples(index=False, name=None))
        predicted_with_epoch = set(self.predicted_pareto[["key_str", "epoch"]].itertuples(index=False, name=None))
        tp_with_epoch = len(ground_truth_with_epoch & predicted_with_epoch)
        fp_with_epoch = len(predicted_with_epoch - ground_truth_with_epoch)
        fn_with_epoch = len(ground_truth_with_epoch - predicted_with_epoch)

        # Loose matching (key_str only)
        ground_truth_ignore_epoch = set(self.ground_truth_pareto["key_str"])
        predicted_ignore_epoch = self.predicted_pareto["key_str"]

        # Frequency count for predicted key_str
        predicted_counts = Counter(predicted_ignore_epoch)

        # Compute weighted true positives, false positives, and false negatives
        tp_ignore_epoch_weighted = sum(predicted_counts[key] for key in ground_truth_ignore_epoch if key in predicted_counts)
        fp_ignore_epoch_weighted = sum(predicted_counts.values()) - tp_ignore_epoch_weighted
        fn_ignore_epoch_weighted = len(ground_truth_ignore_epoch - set(predicted_counts.keys()))

        # Unweighted counts for comparison
        tp_ignore_epoch = len(ground_truth_ignore_epoch & set(predicted_ignore_epoch))
        fp_ignore_epoch = len(set(predicted_ignore_epoch) - ground_truth_ignore_epoch)
        fn_ignore_epoch = len(ground_truth_ignore_epoch - set(predicted_ignore_epoch))

        # Relaxed matching (key_str + epoch within ±5)
        predicted_dict = defaultdict(list)
        for _, row in self.predicted_pareto.iterrows():
            predicted_dict[row["key_str"]].append(row["epoch"])

        tp_relaxed_epoch = 0
        matched_keys = set()
        # Count true positives, allowing multiple matches for relaxed criteria
        for key, true_epoch in self.ground_truth_pareto.set_index("key_str")["epoch"].items():
            if key in predicted_dict:
                # Find all predictions matching this ground truth entry within the tolerance range
                matching_epochs = [pred_epoch for pred_epoch in predicted_dict[key] if abs(pred_epoch - true_epoch) <= 5]
                tp_relaxed_epoch += len(matching_epochs)  # Reward all matches
                if matching_epochs:
                    matched_keys.add(key)  # Mark the ground truth entry as matched

        # Update false negatives: ground truth entries without any matches
        fn_relaxed_epoch = len(self.ground_truth_pareto) - len(matched_keys)

        # Update false positives: unmatched predictions
        fp_relaxed_epoch = sum(len(predicted_dict[key]) for key in predicted_dict) - tp_relaxed_epoch
        fn_relaxed_epoch = len(self.ground_truth_pareto) - len(matched_keys)
        fp_relaxed_epoch = sum(len(predicted_dict[key]) for key in predicted_dict) - tp_relaxed_epoch

        return {
            "with_epoch": {
                **self.compute_metrics(tp_with_epoch, fp_with_epoch, fn_with_epoch),
                "true_positives": tp_with_epoch,
                "false_positives": fp_with_epoch,
                "false_negatives": fn_with_epoch,
            },
            "ignore_epoch": {
                **self.compute_metrics(tp_ignore_epoch_weighted, fp_ignore_epoch_weighted, fn_ignore_epoch_weighted),
                "true_positives": tp_ignore_epoch_weighted,
                "false_positives": fp_ignore_epoch_weighted,
                "false_negatives": fn_ignore_epoch_weighted,
                #"unweighted_metrics": self.compute_metrics(tp_ignore_epoch, fp_ignore_epoch, fn_ignore_epoch),
            },
            "relaxed_epoch": {
                **self.compute_metrics(tp_relaxed_epoch, fp_relaxed_epoch, fn_relaxed_epoch),
                "true_positives": tp_relaxed_epoch,
                "false_positives": fp_relaxed_epoch,
                "false_negatives": fn_relaxed_epoch,
            }
        }

    def compute_ndcg(self, pareto_solutions, acc_weight, em_weight):
        """
        Compute the Normalized Discounted Cumulative Gain (NDCG) using user-defined weights.
        """
        # Compute weighted sorting and relevance metrics
        pareto_solutions["weighted_sorting"] = (
            acc_weight * pareto_solutions["predicted_ACC"] + em_weight * (1 - pareto_solutions["predicted_EM"])
        )
        pareto_solutions["weighted_relevance"] = (
            acc_weight * pareto_solutions["true_ACC"] + em_weight * (1 - pareto_solutions["true_EM"])
        )

        ranked_solutions = pareto_solutions.sort_values(by="weighted_sorting", ascending=False)
        relevance_scores = ranked_solutions["weighted_relevance"].values

        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(sorted(relevance_scores, reverse=True)))

        return dcg / idcg if idcg > 0 else 0

    def get_ndcg_sensitivity(self, weight_range):
        """
        Analyze NDCG sensitivity to varying weights.
        """
        ndcg_results = []
        for acc_w in weight_range:
            em_w = 1 - acc_w
            ndcg = self.compute_ndcg(self.predicted_pareto, acc_weight=acc_w, em_weight=em_w)
            ndcg_results.append({"acc_w": acc_w, "em_w": em_w, "ndcg": ndcg})

        return pd.DataFrame(ndcg_results)

    def hausdorff_distance(self):
        """
        Compute Hausdorff distance between two sets of Pareto points.
        """
        distances = cdist(self.true_pareto_set, self.predicted_pareto_set)  # Pairwise distances
        forward = distances.min(axis=1).max()  # sup_{a in A} inf_{b in B}
        backward = distances.min(axis=0).max()  # sup_{b in B} inf_{a in A}
        return max(forward, backward)

    def hypervolume_difference(self):
        """
        Compute Hypervolume difference between two Pareto frontiers.
        """
        reference_point = np.array([0.0, 1.0])  #worst case 0 accuracy, 1 emissions
        hv = HV(ref_point=reference_point)
        hv_true = hv.do(self.true_pareto_set)  # Hypervolume of true frontier
        hv_pred = hv.do(self.predicted_pareto_set)  # Hypervolume of predicted frontier
        return hv_true - hv_pred

    

class RankingSolutions:
    def __init__(self, pareto_data: dict, weights: list, maximize: list, disagreement_point: list, ideal_point: list, min_acc: float = 0.7):
        """
        Initialize the ranking solution with separated Pareto data.

        Args:
            pareto_data (dict): A dictionary containing Pareto DataFrames (e.g., {"ground_truth": df1, "predicted": df2}).
            weights (list): Weights for objectives (e.g., [0.7, 0.3]).
            maximize (list): Boolean list indicating whether to maximize each objective.
            disagreement_point (list): The disagreement point for Nash Bargaining Solution.
            ideal_point (list): The ideal point for distance-based ranking.
            min_acc (float): Minimum accuracy filter for ranking.
        """
        self.pareto_data = pareto_data
        self.weights = weights
        self.maximize = maximize
        self.disagreement_point = disagreement_point
        self.ideal_point = ideal_point
        self.min_acc = min_acc

        # Define objective labels for each Pareto set
        self.objective_labels = {
            "ground_truth": ["true_ACC", "true_EM"],
            "predicted": ["predicted_ACC", "predicted_EM"]
        }

    def filter_and_normalize_objectives(self, df: pd.DataFrame, objectives: list) -> pd.DataFrame:
            """
            Filter and normalize objectives for scoring within a Pareto set.

            Args:
                df (pd.DataFrame): DataFrame containing Pareto-optimal solutions.
                objectives (list): List of objective columns to normalize.

            Returns:
                pd.DataFrame: DataFrame with normalized objectives.
            """
            filtered_df = df[df[objectives[0]] >= self.min_acc] if objectives[0] else df
            normalized = pd.DataFrame()
            for obj, maximize_flag in zip(objectives, self.maximize):
                if maximize_flag:
                    normalized[obj] = (filtered_df[obj] - filtered_df[obj].min()) / (filtered_df[obj].max() - filtered_df[obj].min())
                else:
                    normalized[obj] = (filtered_df[obj].max() - filtered_df[obj]) / (filtered_df[obj].max() - filtered_df[obj].min())
            return normalized

   
    def rank_by_weighted_scoring(self, df: pd.DataFrame, objectives: list) -> pd.DataFrame:
        """
        Rank Pareto solutions using weighted scoring.

        Args:
            df (pd.DataFrame): DataFrame containing Pareto-optimal solutions.
            objectives (list): List of objective columns to consider.

        Returns:
            pd.DataFrame: DataFrame sorted by weighted scores.
        """
        normalized = self.filter_and_normalize_objectives(df, objectives)
        scores = normalized.mul(self.weights, axis=1).sum(axis=1)
        return df.assign(Score=scores).sort_values("Score", ascending=False)

    def rank_by_nbs(self, df: pd.DataFrame, objectives: list) -> pd.DataFrame:
        """
        Rank Pareto solutions using Nash Bargaining Solution (NBS).

        Args:
            df (pd.DataFrame): DataFrame containing Pareto-optimal solutions.
            objectives (list): List of objective columns to consider.

        Returns:
            pd.DataFrame: DataFrame sorted by NBS scores.
        """
        gains = df[objectives].values - self.disagreement_point
        nbs_scores = np.prod(np.maximum(gains, 0), axis=1)
        return df.assign(NBS_Score=nbs_scores).sort_values("NBS_Score", ascending=False)

    def rank_by_distance_to_ideal(self, df: pd.DataFrame, objectives: list) -> pd.DataFrame:
        """
        Rank Pareto solutions by distance to the ideal point.

        Args:
            df (pd.DataFrame): DataFrame containing Pareto-optimal solutions.
            objectives (list): List of objective columns to consider.

        Returns:
            pd.DataFrame: DataFrame sorted by distance to the ideal point.
        """
        distances = np.sqrt(((df[objectives] - self.ideal_point) ** 2).mul(self.weights, axis=1).sum(axis=1))
        return df.assign(Distance=distances).sort_values("Distance")


    def rank_all(self):
        """
        Rank solutions using all methods across all Pareto sets.

        Returns:
            dict: A dictionary of DataFrames containing ranked solutions for each Pareto set.
        """
        results = {
        "ground_truth": {},
        "predicted": {}
         }
        for name, df in self.pareto_data.items():
            print(f"Ranking Pareto set: {name}")
            objectives = self.objective_labels[name]  # Use appropriate labels for each Pareto set
            results[name]["ranked_by_score"] = self.rank_by_weighted_scoring(df, objectives)
            results[name]["ranked_by_nbs"] = self.rank_by_nbs(df, objectives)
            results[name]["ranked_by_distance"] = self.rank_by_distance_to_ideal(df, objectives)
        return results

class EvaluateRankingSolutions:
    def __init__(self, results, metrics_at=np.array([1, 5, 10, 20, 50]), rbo_p=0.9, key_columns=None):
        """
        Args:
        - results (dict): A dictionary with outer keys "ground_truth" and "predicted", and inner keys as ranking methods.
        - metrics_at (numpy array): Positions at which metrics are computed. Default is [1, 5, 10, 20, 50].
        - rbo_p (float): Parameter controlling the weight decay in RBO. Default is 0.9.
        - key_columns (list): List of columns to consider for ranking comparison. Default is None (use all columns).
        """
        self.ground_truth = results["ground_truth"]
        self.predicted = results["predicted"]
        self.metrics_at = metrics_at
        self.rbo_p = rbo_p
        self.key_columns = key_columns or ['model_name', 'bs', 'lr','epoch']

    def compute_metrics_for_all_methods(self):
        """
        Compute metrics (RBO, FRBO, and Jaccard) for all ranking methods.

        Returns:
        - dict: A dictionary with outer keys as ranking methods and inner dictionaries containing metrics.
        """
        metrics_results = {}

        for method in self.ground_truth.keys():
            # Extract the ground truth and predicted rankings for this method
            ground_truth_ranking = self._extract_relevant_columns(self.ground_truth[method])
            predicted_ranking = self._extract_relevant_columns(self.predicted[method])

            # Compute metrics for this ranking pair
            metrics_results[method] = self.compute_rls_metrics_with_frbo(ground_truth_ranking, predicted_ranking)

        return metrics_results

    def _extract_relevant_columns(self, df):
        """
        Extract and convert the specified columns into a list of tuples for ranking comparison.

        Args:
        - df (DataFrame): The input DataFrame containing rankings.

        Returns:
        - list: A list of tuples representing the ranking based on the key columns.
        """
        return df[self.key_columns].apply(tuple, axis=1).tolist()

    def compute_rls_metrics_with_frbo(self, ground_truth, predicted):
        """
        Compute RBO, FRBO, and Jaccard similarity for a single pair of ranked lists.

        Args:
        - ground_truth (list): Ground truth ranked list (list of tuples).
        - predicted (list): Predicted ranked list (list of tuples).

        Returns:
        - dict: A dictionary containing RBO, FRBO, and Jaccard metrics at specified positions.
        """
        rls_rbo = np.zeros(len(self.metrics_at))
        rls_frbo = np.zeros(len(self.metrics_at))
        rls_jac = np.zeros(len(self.metrics_at))

        j = 0
        rbo_sum = 0
        for d in range(1, min(min(len(ground_truth), len(predicted)), max(self.metrics_at)) + 1):
            # Create sets of the first d elements from the two ranked lists
            set_gt, set_pred = set(ground_truth[:d]), set(predicted[:d])

            # Calculate the intersection cardinality of the sets
            inters_card = len(set_gt.intersection(set_pred))

            # Update RBO sum using the formula
            rbo_sum += self.rbo_p ** (d - 1) * inters_card / d
            if d == self.metrics_at[j]:
                # Update RBO at the specified position
                rls_rbo[j] += (1 - self.rbo_p) * rbo_sum / (1 - self.rbo_p ** d)
                
                # Compute FRBO by normalizing RBO
                max_rbo = (1 - self.rbo_p) * np.sum([self.rbo_p ** (i - 1) / i for i in range(1, d + 1)])
                frbo = rls_rbo[j] / max_rbo if max_rbo > 0 else 0
                rls_frbo[j] += frbo

                # Update Jaccard similarity at the specified position
                rls_jac[j] += inters_card / len(set_gt.union(set_pred))
                j += 1

        # Handle cases where lists are shorter than the largest cutoff
        if j != len(self.metrics_at):
            for k in range(j, len(self.metrics_at)):
                rls_rbo[k] += (1 - self.rbo_p) * rbo_sum
                max_rbo = (1 - self.rbo_p) * np.sum([self.rbo_p ** (i - 1) / i for i in range(1, d + 1)])
                frbo = rls_rbo[k] / max_rbo if max_rbo > 0 else 0
                rls_frbo[k] += frbo
                rls_jac[k] += inters_card / len(set_gt.union(set_pred))

        # Normalize and create dictionaries for results
        rbo_dict = {"@"+str(k): rls_rbo[i] for i, k in enumerate(self.metrics_at)}
        frbo_dict = {"@"+str(k): rls_frbo[i] for i, k in enumerate(self.metrics_at)}
        jac_dict = {"@"+str(k): rls_jac[i] for i, k in enumerate(self.metrics_at)}

        return {"RLS_RBO": rbo_dict, "RLS_FRBO": frbo_dict, "RLS_JAC": jac_dict}


if __name__ == "__main__":
    # Load test data
    df= pd.read_csv("results_csv//test_results.csv")
    test_results = df[df['epoch'] <= 100]

    test_results['error_ACC'] = abs(test_results['true_ACC'] - test_results['predicted_ACC'])
    test_results['error_EM'] = abs(test_results['true_EM'] - test_results['predicted_EM'])

    mae_first = np.mean(test_results['error_ACC'] )  # MAE for the first label
    mae_rest = np.mean(test_results['error_EM'] )    # MAE for the second label

    #avg_uncertainty_first = np.mean(std_pred_first)
    #avg_uncertainty_rest = np.mean(std_pred_rest)

    # Print the results
    print(f"MAE for the VAL_ACC label: {mae_first}")
    #print(f"Average Uncertainty for VAL_ACC: {avg_uncertainty_first}")

    print(f"MAE for the EMISSION label: {mae_rest}")


    maximize = [True, False]
    accuracy_threshold = 0.70
    true_objectives = ["true_ACC", "true_EM"]
    predicted_objectives = ["predicted_ACC", "predicted_EM"]


    optimizer = ParetoSolutions(
        data=test_results,
        true_objectives=true_objectives,
        predicted_objectives=predicted_objectives,
        min_acc=accuracy_threshold,
        maximize=maximize
    )
    ground_truth_pareto, predicted_pareto= optimizer.get_pareto_dfs()
    pareto_data = {
    "ground_truth": ground_truth_pareto,
    "predicted": predicted_pareto
    }

    # Debug: Check Pareto fronts
    #print("Ground Truth Pareto Front BEFORE Filtering:")
    #print(optimizer._find_pareto_front(true_objectives)[['model_name', 'bs', 'lr','epoch','true_ACC', 'predicted_ACC', 'true_EM', 'predicted_EM']])

    print("\nGround Truth Pareto Front AFTER Filtering:")
    print(ground_truth_pareto[['model_name', 'bs', 'lr','epoch','true_ACC', 'predicted_ACC', 'true_EM', 'predicted_EM']])

   # print("Predicted Pareto Front BEFORE Filtering:")
    #print(optimizer._find_pareto_front(predicted_objectives)[['model_name', 'bs', 'lr','epoch','true_ACC', 'predicted_ACC', 'true_EM', 'predicted_EM']])
    
    print("\nPredicted Truth Pareto Front AFTER Filtering:")
    print(predicted_pareto[['model_name', 'bs', 'lr','epoch','true_ACC', 'predicted_ACC', 'true_EM', 'predicted_EM']])

    evaluator = EvaluateParetoSolutions(
        ground_truth_pareto=ground_truth_pareto,
        predicted_pareto=predicted_pareto
    )

    # Analyze NDCG sensitivity
    weight_range = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
    ndcg_results = evaluator.get_ndcg_sensitivity(weight_range)
    print(f"NDCG Sensitivity Results:\n{ndcg_results}")

    # Plot NDCG sensitivity
    plot_ndcg_sensitivity(ndcg_results)

    # Compute standard metrics (precision, recall, F1)
    metrics = evaluator.get_standard_metrics()
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")

    #Difference across Pareto sets 
    hausdorff = evaluator.hausdorff_distance()
    hypervolume_diff = evaluator.hypervolume_difference()
    print(f"Hausdorff Distance: {hausdorff}")
    print(f"Hypervolume Difference: {hypervolume_diff}")

    print("\nMatching Metrics:")
    print(metrics_df)
   
    #Plot pareto fronts
    plot_pareto_fronts(ground_truth_pareto, predicted_pareto)
    

    weights = [0.7, 0.3] #example of weigths
    disagreement_point = np.array([accuracy_threshold, 1])  # Worst-case values
    ideal_point = np.array([1, 0])  # Best possible values


    # Initialize RankingSolutions class
    ranking = RankingSolutions(
        pareto_data=pareto_data,
        weights=[0.7, 0.3],
        maximize=[True, False],
        disagreement_point=[0.7, 1],
        ideal_point=[1, 0],
        min_acc=0.7
    )

    # Rank solutions using all methods
    results = ranking.rank_all()
    print(results)

    for name, metrics in results.items():
        print(f"Ranked results for {name}")
        for metric, df in metrics.items():
            print(f"{metric} :\n{df[['model_name', 'bs', 'lr','epoch','true_ACC', 'predicted_ACC', 'true_EM', 'predicted_EM']]}")

    ranking_evaluator = EvaluateRankingSolutions(results)

    # Compute metrics for all methods
    metrics = ranking_evaluator.compute_metrics_for_all_methods()

    # Display results
    print(metrics)