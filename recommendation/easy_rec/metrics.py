import torch
import torchmetrics


def prepare_rank_corrections(metrics_info, num_negatives = None, num_items = None, put_uncorrected = True, split_keys={"train":1,"val":2,"test":1}):
    metrics = {}

    # This part could be improved by using a function
    if isinstance(metrics_info, dict) and all([key in metrics_info for key in split_keys.keys()]):
        metrics_info_already_split = True
    else:
        metrics_info_already_split = False
    if num_negatives is not None:
        if isinstance(num_negatives, dict) and all([key in num_negatives for key in split_keys.keys()]):
            num_negatives_already_split = True
        else:
            num_negatives_already_split = False
    if num_items is not None:
        if isinstance(num_items, dict) and all([key in num_items for key in split_keys.keys()]):
            num_items_already_split = True
        else:
            num_items_already_split = False

    if isinstance(put_uncorrected, dict) and all([key in put_uncorrected for key in split_keys.keys()]):
        put_uncorrected_already_split = True
    else:
        put_uncorrected_already_split = False

    for split_name, num_dataloaders in split_keys.items():
        metrics[split_name] = []
        for dataloader_idx in range(num_dataloaders):
            metrics[split_name].append({})

            if metrics_info_already_split:
                metrics_info_to_use = metrics_info[split_name][dataloader_idx]
            else:
                metrics_info_to_use = metrics_info
    
            for metric_name in metrics_info_to_use:
                if isinstance(metrics_info_to_use, list): 
                    metric_vals = {}  # Initialize an empty dictionary for metric parameters
                elif isinstance(metrics_info_to_use, dict): 
                    metric_vals = metrics_info_to_use[metric_name]  # Get metric parameters from the provided dictionary
                else: 
                    raise NotImplementedError  # Raise an error for unsupported input types
                metrics[split_name][dataloader_idx][metric_name] = metric_vals

                if metric_name in ["RLS_Jaccard", "RLS_RBO", "RLS_FRBO",
                                "NDCG", "MRR", "Precision", "Recall", "F1", "PrecisionWithRelevance", "MAP"]:
                    rank_corrections = {}
                    if put_uncorrected_already_split:
                        put_uncorrected_to_use = put_uncorrected[split_name][dataloader_idx]
                    else:
                        put_uncorrected_to_use = put_uncorrected
                    if put_uncorrected_to_use:
                        rank_corrections[""] = lambda x: x
                    if num_negatives_already_split:
                        num_negatives_to_use = num_negatives[split_name][dataloader_idx]
                    else:
                        num_negatives_to_use = num_negatives
                    if num_items_already_split:
                        num_items_to_use = num_items[split_name][dataloader_idx]
                    else:
                        num_items_to_use = num_items
                    if num_negatives_to_use is not None and num_items_to_use is not None:
                        correction = num_items_to_use/num_negatives_to_use
                        rank_corrections["corrected"] = lambda x: correction * x
            
                    metrics[split_name][dataloader_idx][metric_name]["rank_corrections"] = rank_corrections

    return metrics

class RecMetric(torchmetrics.Metric):
    """
        Initializes the RecMetric.

        Args:
            top_k (list): List of integers representing top-k values for evaluation.
            batch_metric (bool): Whether to compute metrics on batch level or not.
    """
    def __init__(self, top_k = [5,10,20], batch_metric = False, rank_corrections = {"": lambda x: x}):
        super().__init__()
        self.top_k = top_k if isinstance(top_k, list) else [top_k]
        self.batch_metric = batch_metric
        self.rank_corrections = rank_corrections

        # Initialize state variables for correct predictions and total examples
        for rank_correction_name in self.rank_corrections.keys():
            for top_k in self.top_k:
                if not self.batch_metric:
                    self.add_state(f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", default=torch.tensor(0.), dist_reduce_fx="sum")
                else:
                    self.add_state(f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", default=[], dist_reduce_fx="cat")
        
        if not self.batch_metric:
            self.add_state(f"total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def compute(self):
        """
        Computes and returns the metric values.

        Returns:
            dict: Dictionary containing metric values.
        """
        # Compute accuracy as the ratio of correct predictions to total examples
        out = {}
        for rank_correction_name in self.rank_corrections.keys():
            for k in self.top_k:
                out[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"] = getattr(self, f"correct@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}")
                if not self.batch_metric:
                    out[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"] = out[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"] / self.total
                else:
                    out[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"] = torchmetrics.utilities.dim_zero_cat(out[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"])
        return out
    
    def not_nan_subset(self, **kwargs):
        """
        Subsets input tensors where the 'relevance' tensor is not NaN.

        Returns:
            dict: Subset of input tensors where 'relevance' is not NaN.
        """
        if "relevance" in kwargs:
            # Subset other args, kwargs where relevance is not nan
            relevance = kwargs["relevance"]
            app = torch.isnan(relevance)
            is_not_nan_per_sample = ~app.all(-1)
            kwargs = {k: v[is_not_nan_per_sample] for k, v in kwargs.items()}
            kwargs["relevance"][app[is_not_nan_per_sample]] = 0
            # This keeps just the last dimension, the others are collapsed

        return kwargs
    
#TODO: check RLS after rank_correction
class RLS_Jaccard(RecMetric):
    '''
     ...
    '''
    def __init__(self, rbo_p=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rbo_p = rbo_p

    def update(self, scores: torch.Tensor, other_scores: torch.Tensor, relevance: torch.Tensor):
        """
        Updates the metric values based on the input scores and relevance tensors.

        Args:
            scores (torch.Tensor): Tensor containing prediction scores.
            other_scores (torch.Tensor): Tensor containing other prediction scores.
            relevance (torch.Tensor): Tensor containing relevance values.
        """

        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, other_scores=other_scores, relevance=relevance)
        scores, other_scores, relevance = kwargs["scores"], kwargs["other_scores"], kwargs["relevance"]
        
        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1
        other_ordered_items = other_scores.argsort(dim=-1, descending=True)
        other_ranks = other_ordered_items.argsort(dim=-1)+1
        
        for rank_correction_name,rank_correction_function in self.rank_corrections.items():
            correct_ranks = rank_correction_function(ranks)
            correct_other_ranks = rank_correction_function(other_ranks)
            for top_k in self.top_k:
                app1, app2 = correct_ranks<=top_k, correct_other_ranks<=top_k
                intersection_size = torch.logical_and(app1, app2).sum(-1)
                union_size = torch.logical_or(app1, app2).sum(-1)
                jaccard_sim = intersection_size.float()/union_size
                if not self.batch_metric:
                    setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}") + jaccard_sim.sum())
                else:
                    getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}").append(jaccard_sim)
        
        if not self.batch_metric:
            self.total += relevance.shape[0]

class RLS_RBO(RecMetric):
    '''
     TODO:...
    '''
    def __init__(self, rbo_p=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rbo_p = rbo_p

    def update(self, scores: torch.Tensor, other_scores: torch.Tensor, relevance: torch.Tensor):
        """
        Updates the metric values based on the input scores and relevance tensors.

        Args:
            scores (torch.Tensor): Tensor containing prediction scores.
            other_scores (torch.Tensor): Tensor containing other prediction scores.
            relevance (torch.Tensor): Tensor containing relevance values.
        """

        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, other_scores=other_scores, relevance=relevance)
        scores, other_scores, relevance = kwargs["scores"], kwargs["other_scores"], kwargs["relevance"]
        
        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1
        other_ordered_items = other_scores.argsort(dim=-1, descending=True)
        other_ranks = other_ordered_items.argsort(dim=-1)+1
        
        for rank_correction_name,rank_correction_function in self.rank_corrections.items():
            correct_ranks = rank_correction_function(ranks)
            correct_other_ranks = rank_correction_function(other_ranks)

            intersection_sizes_sum = torch.zeros(correct_ranks.shape[0], device=relevance.device, dtype=torch.float32)
            for top_k in range(1,max(self.top_k)+1):
                app1, app2 = correct_ranks<=top_k, correct_other_ranks<=top_k
                intersection_size = torch.logical_and(app1, app2).sum(-1)
                intersection_sizes_sum += (self.rbo_p**(top_k-1))*intersection_size/top_k
                if top_k in self.top_k:
                    rbo = (1-self.rbo_p)*intersection_sizes_sum
                    if not self.batch_metric:
                        setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}") + rbo.sum())
                    else:
                        getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}").append(rbo)
        
        if not self.batch_metric:
            self.total += relevance.shape[0]

class RLS_FRBO(RecMetric):
    '''
     ...
    '''
    def __init__(self, rbo_p=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rbo_p = rbo_p

    def update(self, scores: torch.Tensor,  other_scores: torch.Tensor, relevance: torch.Tensor):
        """
        Updates the metric values based on the input scores and relevance tensors.

        Args:
            scores (torch.Tensor): Tensor containing prediction scores.
            other_scores (torch.Tensor): Tensor containing other prediction scores.
            relevance (torch.Tensor): Tensor containing relevance values.
        """

        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, other_scores=other_scores, relevance=relevance)
        scores, other_scores, relevance = kwargs["scores"], kwargs["other_scores"], kwargs["relevance"]
        
        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1
        other_ordered_items = other_scores.argsort(dim=-1, descending=True)
        other_ranks = other_ordered_items.argsort(dim=-1)+1
        
        for rank_correction_name,rank_correction_function in self.rank_corrections.items():
            correct_ranks = rank_correction_function(ranks)
            correct_other_ranks = rank_correction_function(other_ranks)

            intersection_sizes_sum = torch.zeros(correct_ranks.shape[0], device=relevance.device, dtype=torch.float32)
            for top_k in range(1,max(self.top_k)+1):
                app1, app2 = correct_ranks<=top_k, correct_other_ranks<=top_k
                intersection_size = torch.logical_and(app1, app2).sum(-1)
                intersection_sizes_sum += (self.rbo_p**(top_k-1))*intersection_size/top_k
                if top_k in self.top_k:
                    frbo = (1-self.rbo_p)/(1-(self.rbo_p**top_k))*intersection_sizes_sum
                    if not self.batch_metric:
                        setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}") + frbo.sum())
                    else:
                        getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}").append(frbo)
        
        if not self.batch_metric:
            self.total += relevance.shape[0]

# def compute_rls_metrics(preds1, preds2, metrics_at=np.array([1,5,10,20,50]), rbo_p=0.9):
#     """
#     Compute Rank-Biased Overlap (RBO) and Jaccard similarity metrics for ranked lists.

#     Args:
#     - preds1 (list): List of ranked predictions (e.g., recommendations).
#     - preds2 (list): List of ranked predictions for comparison.
#     - metrics_at (numpy array, optional): Positions at which metrics are computed. Default is [1, 5, 10, 20, 50].
#     - rbo_p (float, optional): Parameter controlling the weight decay in RBO. Default is 0.9.

#     Returns:
#     - dict: A dictionary containing RBO and Jaccard metrics at specified positions.
#     """
#     # Initialize arrays to store RBO and Jaccard scores at specified positions
#     rls_rbo = np.zeros((len(metrics_at)))
#     rls_jac = np.zeros((len(metrics_at)))

#     for pred1,pred2 in zip(preds1,preds2):
#         j = 0
#         rbo_sum = 0
#         for d in range(1,min(min(len(pred1),len(pred2)),max(metrics_at))+1):
#             # Create sets of the first d elements from the two ranked lists
#             set_pred1, set_pred2 = set(pred1[:d]), set(pred2[:d])
            
#             # Calculate the intersection cardinality of the sets
#             inters_card = len(set_pred1.intersection(set_pred2))

#             # Update RBO sum using the formula
#             rbo_sum += rbo_p**(d-1)*inters_card/d
#             if d==metrics_at[j]:
#                 # Update RBO and Jaccard scores at the specified position
#                 rls_rbo[j] += (1-rbo_p)*rbo_sum/(1-rbo_p**d)
                        
#                 rls_jac[j] += inters_card/len(set_pred1.union(set_pred2))
#                 # Move to the next specified position
#                 j+=1
#         #Check if it has stopped before cause pred1 or pred2 are shorter
#         if j!=len(metrics_at):
#             for k in range(j,len(metrics_at)):
#                 rls_rbo[k] += (1-rbo_p)*rbo_sum
#                 rls_jac[k] += inters_card/len(set_pred1.union(set_pred2))
#     # Create dictionaries with specified positions as keys and normalized scores            
#     rbo_dict = {"@"+str(k):rls_rbo[i]/len(preds1) for i,k in enumerate(metrics_at)}
#     jac_dict = {"@"+str(k):rls_jac[i]/len(preds1) for i,k in enumerate(metrics_at)}
#     # Return a dictionary containing RBO and Jaccard results
#     return {"RLS_RBO":rbo_dict, "RLS_JAC":jac_dict}
    
class NDCG(RecMetric):
    '''
     Normalized Discounted Cumulative Gain (NDCG) assesses the performance of a ranking system by considering the placement of K relevant items 
     within the ranked list. The underlying principle is that items higher in the ranking should receive a higher score than those positioned 
     lower in the list because they are those where a user's attention is usually focused.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return super().compute()

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        """
        Updates the metric values based on the input scores and relevance tensors.

        Args:
            scores (torch.Tensor): Tensor containing prediction scores.
            relevance (torch.Tensor): Tensor containing relevance values.
        """
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        for rank_correction_name,rank_correction_function in self.rank_corrections.items():
            correct_ranks = rank_correction_function(ranks)
            app = torch.log2(correct_ranks+1)
            for top_k in self.top_k:
                dcg = ((correct_ranks<=top_k)*relevance/app).sum(-1)
                k = min(top_k,scores.shape[-1])
                sorted_k_relevance = relevance.sort(dim=-1, descending=True).values[...,:k] #get first k items in sorted_relevance on last dimension  
                idcg = (sorted_k_relevance/torch.log2(torch.arange(1,k+1,device=sorted_k_relevance.device)+1)).sum(-1)
                ndcg = dcg/idcg # ndcg.shape = (num_samples, lookback)
                if not self.batch_metric:
                    setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}") + ndcg.sum())
                else:
                    getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}").append(ndcg)
        if not self.batch_metric:
            self.total += relevance.shape[0]
    
class MRR(RecMetric):
    '''
    Mean Reciprocal Rank (MRR) evaluates the efficacy of a ranking system by considering the placement of the first relevant item within the ranked list.
    It is calculated by taking the reciprocal of the rank of the first relevant item.
    It emphasizes that the position of the first relevant item is more important than the placement of the other relevant items.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for rank_correction_name,rank_correction_function in self.rank_corrections.items():
            correct_ranks = rank_correction_function(ranks)
            for top_k in self.top_k:
                mrr = ((correct_ranks<=top_k)*relevant*(1/correct_ranks)).max(-1).values
                if not self.batch_metric:
                    setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}") + mrr.sum())
                else:
                    getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}").append(mrr)
        if not self.batch_metric:
            self.total += relevance.shape[0]

class Precision(RecMetric):
    '''
    It computes the proportion of accurately identified relevant items among all the items recommended within a list of length K. 
    It is used to explicitly count the number of recommended, or retrieved, items that are truly relevant.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        """
        Updates the metric values based on the input scores and relevance tensors.

        Args:
            scores (torch.Tensor): Tensor containing prediction scores.
            relevance (torch.Tensor): Tensor containing relevance values.
        """
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for rank_correction_name,rank_correction_function in self.rank_corrections.items():
            correct_ranks = rank_correction_function(ranks)
            for top_k in self.top_k:
                precision = ((correct_ranks<=top_k)*relevant/top_k).sum(-1)
                if not self.batch_metric:
                    setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}") + precision.sum())
                else:
                    getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}").append(precision)
        if not self.batch_metric:
            self.total += relevance.shape[0]

class Recall(RecMetric):
    '''
    It assesses the fraction of correctly identified relevant items among the top K recommendations, relative to the total number of relevant items in the dataset. 
    It measures the effectiveness of the method in capturing relevant items among all of those present in the dataset.
    
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for rank_correction_name,rank_correction_function in self.rank_corrections.items():
            correct_ranks = rank_correction_function(ranks)
            for top_k in self.top_k:
                recall = ((correct_ranks<=top_k)*relevant/relevant.sum(-1,keepdim=True)).sum(-1)
                #torch.minimum(relevant.sum(-1,keepdim=True),top_k*torch.ones_like(relevant.sum(-1,keepdim=True)))
                if not self.batch_metric:
                    setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}") + recall.sum())
                else:
                    getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}").append(recall)
        if not self.batch_metric:
            self.total += relevance.shape[0]

class F1(RecMetric):
    '''
    The F1 score is the harmonic mean of precision and recall. 
    It is a single metric that combines both precision and recall to provide a single measure of the quality of a ranking system.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = Precision(*args, **kwargs)
        self.recall = Recall(*args, **kwargs)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        self.precision.update(scores, relevance)
        self.recall.update(scores, relevance)

    def compute(self):
        precision = self.precision.compute()
        recall = self.recall.compute()
        out = {}
        for rank_correction_name in self.rank_corrections.keys():
            for k in self.top_k:
                out[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"] = 2*(precision[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"]*recall[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"])/(precision[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"]+recall[f"@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"])
        return out

class PrecisionWithRelevance(RecMetric):
    '''
    It computes the proportion of accurately identified relevant items among all the items recommended within a list of length K.
    It is used to explicitly count the number of recommended, or retrieved, items that are truly relevant.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        for rank_correction_name,rank_correction_function in self.rank_corrections.items():
            correct_ranks = rank_correction_function(ranks)
            
            for top_k in self.top_k:
                precision = ((correct_ranks<=top_k)*relevance/(top_k*relevance.sum(-1,keepdim=True))).sum(-1)
                if not self.batch_metric:
                    setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}") + precision.sum())
                else:
                    getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}").append(precision)

        if not self.batch_metric:
            self.total = self.total + relevance.shape[0] #not using += cause getting InferenceMode error sometimes

class MAP(RecMetric):
    '''
    Mean Average Precision (MAP) evaluates the efficacy of a ranking system by considering the average precision across the top R recommendations for R ranging from 1 to K. 
    It emphasizes that precision values for items within the top K positions contribute to the overall assessment also accounting for the significance of the order in the ranking. 
    Different from NDCG, this metric does not explicitly assign a different importance to different slots.
    
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.precision_at_k = PrecisionWithRelevance(list(range(1,torch.max(torch.tensor(self.top_k))+1)), self.batch_metric, self.rank_corrections)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        self.precision_at_k.update(scores, relevance)

    def compute(self):
        if not self.batch_metric:
            for rank_correction_name in self.rank_corrections.keys():
                for top_k in self.top_k:
                    for k in range(1,top_k+1):
                        setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}") + getattr(self.precision_at_k, f"correct@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}"))
                    setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", getattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}")/top_k)
            setattr(self,"total", getattr(self.precision_at_k, f"total"))
        else:
            for rank_correction_name in self.rank_corrections.keys():
                for top_k in self.top_k:
                    correct = getattr(self.precision_at_k, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}")
                    for k in range(1,top_k):
                        new_correct = getattr(self.precision_at_k, f"correct@{'_'.join([str(x) for x in [k,rank_correction_name] if x])}")
                        for i,c in enumerate(new_correct):
                            correct[i] += c
                    setattr(self, f"correct@{'_'.join([str(x) for x in [top_k,rank_correction_name] if x])}", [c/top_k for c in correct])
        
        return super().compute()
    
    def reset(self):
        super().reset()
        self.precision_at_k.reset()
















# import torch
# import torchmetrics

# class RecMetric(torchmetrics.Metric):
#     """
#         Initializes the RecMetric.

#         Args:
#             top_k (list): List of integers representing top-k values for evaluation.
#             batch_metric (bool): Whether to compute metrics on batch level or not.
#     """
#     def __init__(self, top_k=[5,10,20], batch_metric = False):
#         super().__init__()
#         self.top_k = top_k if isinstance(top_k, list) else [top_k]
#         self.batch_metric = batch_metric

#         # Initialize state variables for correct predictions and total examples
#         for top_k in self.top_k:
#             if not self.batch_metric:
#                 self.add_state(f"correct@{top_k}", default=torch.tensor(0.), dist_reduce_fx="sum")
#             else:
#                 self.add_state(f"correct@{top_k}", default=[], dist_reduce_fx="cat")
        
#         if not self.batch_metric:
#             self.add_state(f"total", default=torch.tensor(0.), dist_reduce_fx="sum")

#     def compute(self):
#         """
#         Computes and returns the metric values.

#         Returns:
#             dict: Dictionary containing metric values.
#         """
#         # Compute accuracy as the ratio of correct predictions to total examples
#         out = {}
#         for k in self.top_k:
#             out[f"@{k}"] = getattr(self, f"correct@{k}")
#             if not self.batch_metric:
#                 out[f"@{k}"] = out[f"@{k}"] / self.total
#             else:
#                 out[f"@{k}"] = torchmetrics.utilities.dim_zero_cat(out[f"@{k}"])
#         return out
    
#     def not_nan_subset(self, **kwargs):
#         """
#         Subsets input tensors where the 'relevance' tensor is not NaN.

#         Returns:
#             dict: Subset of input tensors where 'relevance' is not NaN.
#         """
#         if "relevance" in kwargs:
#             # Subset other args, kwargs where relevance is not nan
#             relevance = kwargs["relevance"]
#             app = torch.isnan(relevance)
#             is_not_nan_per_sample = ~app.all(-1)
#             kwargs = {k: v[is_not_nan_per_sample] for k, v in kwargs.items()}
#             kwargs["relevance"][app[is_not_nan_per_sample]] = 0
#             # This keeps just the last dimension, the others are collapsed

#         return kwargs
    
# class RLS_Jaccard(RecMetric):
#     '''
#      ...
#     '''
#     def __init__(self, rbo_p=0.9, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.batch_metric=batch_metric
#         self.rbo_p = rbo_p

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         """
#         Updates the metric values based on the input scores and relevance tensors.

#         Args:
#             scores (torch.Tensor): Tensor containing prediction scores.
#             other_scores (torch.Tensor): Tensor containing other prediction scores.
#             relevance (torch.Tensor): Tensor containing relevance values.
#         """

#         #!!!!!CHANGE!!!!!
#         other_scores = scores[torch.arange(scores.shape[0]-1,-1,-1)]

#         # Call not_nan_subset to subset scores, relevance where relevance is not nan
#         kwargs = self.not_nan_subset(scores=scores, other_scores=other_scores, relevance=relevance)
#         scores, other_scores, relevance = kwargs["scores"], kwargs["other_scores"], kwargs["relevance"]
        
#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1
#         other_ordered_items = other_scores.argsort(dim=-1, descending=True)
#         other_ranks = other_ordered_items.argsort(dim=-1)+1
        
#         for top_k in self.top_k:
#             app1, app2 = ranks<=top_k, other_ranks<=top_k
#             intersection_size = torch.logical_and(app1, app2).sum(-1)
#             union_size = torch.logical_or(app1, app2).sum(-1)
#             jaccard_sim = intersection_size.float()/union_size
#             if not self.batch_metric:
#                 setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + jaccard_sim.sum())
#             else:
#                 getattr(self, f"correct@{top_k}").append(jaccard_sim)
        
#         if not self.batch_metric:
#             self.total += relevance.shape[0]

# class RLS_RBO(RecMetric):
#     '''
#      TODO:...
#     '''
#     def __init__(self, rbo_p=0.9, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.batch_metric=batch_metric
#         self.rbo_p = rbo_p

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         """
#         Updates the metric values based on the input scores and relevance tensors.

#         Args:
#             scores (torch.Tensor): Tensor containing prediction scores.
#             other_scores (torch.Tensor): Tensor containing other prediction scores.
#             relevance (torch.Tensor): Tensor containing relevance values.
#         """

#         #!!!!!CHANGE!!!!!
#         other_scores = scores[torch.arange(scores.shape[0]-1,-1,-1)]

#         # Call not_nan_subset to subset scores, relevance where relevance is not nan
#         kwargs = self.not_nan_subset(scores=scores, other_scores=other_scores, relevance=relevance)
#         scores, other_scores, relevance = kwargs["scores"], kwargs["other_scores"], kwargs["relevance"]
        
#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1
#         other_ordered_items = other_scores.argsort(dim=-1, descending=True)
#         other_ranks = other_ordered_items.argsort(dim=-1)+1
        
#         intersection_sizes_sum = torch.zeros(ranks.shape[0], device=relevance.device, dtype=torch.float32)
#         for top_k in range(1,max(self.top_k)+1):
#             app1, app2 = ranks<=top_k, other_ranks<=top_k
#             intersection_size = torch.logical_and(app1, app2).sum(-1)
#             intersection_sizes_sum += (self.rbo_p**(top_k-1))*intersection_size/top_k
#             if top_k in self.top_k:
#                 rbo = (1-self.rbo_p)*intersection_sizes_sum
#                 if not self.batch_metric:
#                     setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + rbo.sum())
#                 else:
#                     getattr(self, f"correct@{top_k}").append(rbo)
        
#         if not self.batch_metric:
#             self.total += relevance.shape[0]

# class RLS_FRBO(RecMetric):
#     '''
#      ...
#     '''
#     def __init__(self, rbo_p=0.9, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.batch_metric=batch_metric
#         self.rbo_p = rbo_p

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         """
#         Updates the metric values based on the input scores and relevance tensors.

#         Args:
#             scores (torch.Tensor): Tensor containing prediction scores.
#             other_scores (torch.Tensor): Tensor containing other prediction scores.
#             relevance (torch.Tensor): Tensor containing relevance values.
#         """

#         #!!!!!CHANGE!!!!!
#         other_scores = scores[torch.arange(scores.shape[0]-1,-1,-1)]

#         # Call not_nan_subset to subset scores, relevance where relevance is not nan
#         kwargs = self.not_nan_subset(scores=scores, other_scores=other_scores, relevance=relevance)
#         scores, other_scores, relevance = kwargs["scores"], kwargs["other_scores"], kwargs["relevance"]
        
#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1
#         other_ordered_items = other_scores.argsort(dim=-1, descending=True)
#         other_ranks = other_ordered_items.argsort(dim=-1)+1
        
#         intersection_sizes_sum = torch.zeros(ranks.shape[0], device=relevance.device, dtype=torch.float32)
#         for top_k in range(1,max(self.top_k)+1):
#             app1, app2 = ranks<=top_k, other_ranks<=top_k
#             intersection_size = torch.logical_and(app1, app2).sum(-1)
#             intersection_sizes_sum += (self.rbo_p**(top_k-1))*intersection_size/top_k
#             if top_k in self.top_k:
#                 frbo = (1-self.rbo_p)/(1-(self.rbo_p**top_k))*intersection_sizes_sum
#                 if not self.batch_metric:
#                     setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + frbo.sum())
#                 else:
#                     getattr(self, f"correct@{top_k}").append(frbo)
        
#         if not self.batch_metric:
#             self.total += relevance.shape[0]

# # def compute_rls_metrics(preds1, preds2, metrics_at=np.array([1,5,10,20,50]), rbo_p=0.9):
# #     """
# #     Compute Rank-Biased Overlap (RBO) and Jaccard similarity metrics for ranked lists.

# #     Args:
# #     - preds1 (list): List of ranked predictions (e.g., recommendations).
# #     - preds2 (list): List of ranked predictions for comparison.
# #     - metrics_at (numpy array, optional): Positions at which metrics are computed. Default is [1, 5, 10, 20, 50].
# #     - rbo_p (float, optional): Parameter controlling the weight decay in RBO. Default is 0.9.

# #     Returns:
# #     - dict: A dictionary containing RBO and Jaccard metrics at specified positions.
# #     """
# #     # Initialize arrays to store RBO and Jaccard scores at specified positions
# #     rls_rbo = np.zeros((len(metrics_at)))
# #     rls_jac = np.zeros((len(metrics_at)))

# #     for pred1,pred2 in zip(preds1,preds2):
# #         j = 0
# #         rbo_sum = 0
# #         for d in range(1,min(min(len(pred1),len(pred2)),max(metrics_at))+1):
# #             # Create sets of the first d elements from the two ranked lists
# #             set_pred1, set_pred2 = set(pred1[:d]), set(pred2[:d])
            
# #             # Calculate the intersection cardinality of the sets
# #             inters_card = len(set_pred1.intersection(set_pred2))

# #             # Update RBO sum using the formula
# #             rbo_sum += rbo_p**(d-1)*inters_card/d
# #             if d==metrics_at[j]:
# #                 # Update RBO and Jaccard scores at the specified position
# #                 rls_rbo[j] += (1-rbo_p)*rbo_sum/(1-rbo_p**d)
                        
# #                 rls_jac[j] += inters_card/len(set_pred1.union(set_pred2))
# #                 # Move to the next specified position
# #                 j+=1
# #         #Check if it has stopped before cause pred1 or pred2 are shorter
# #         if j!=len(metrics_at):
# #             for k in range(j,len(metrics_at)):
# #                 rls_rbo[k] += (1-rbo_p)*rbo_sum
# #                 rls_jac[k] += inters_card/len(set_pred1.union(set_pred2))
# #     # Create dictionaries with specified positions as keys and normalized scores            
# #     rbo_dict = {"@"+str(k):rls_rbo[i]/len(preds1) for i,k in enumerate(metrics_at)}
# #     jac_dict = {"@"+str(k):rls_jac[i]/len(preds1) for i,k in enumerate(metrics_at)}
# #     # Return a dictionary containing RBO and Jaccard results
# #     return {"RLS_RBO":rbo_dict, "RLS_JAC":jac_dict}

    
# class NDCG(RecMetric):
#     '''
#      Normalized Discounted Cumulative Gain (NDCG) assesses the performance of a ranking system by considering the placement of K relevant items 
#      within the ranked list. The underlying principle is that items higher in the ranking should receive a higher score than those positioned 
#      lower in the list because they are those where a user's attention is usually focused.
#     '''
#     def __init__(self, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.batch_metric=batch_metric

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         """
#         Updates the metric values based on the input scores and relevance tensors.

#         Args:
#             scores (torch.Tensor): Tensor containing prediction scores.
#             relevance (torch.Tensor): Tensor containing relevance values.

#         """
#         # Call not_nan_subset to subset scores, relevance where relevance is not nan
#         kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
#         scores, relevance = kwargs["scores"], kwargs["relevance"]

#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1

#         app = torch.log2(ranks+1)
#         for top_k in self.top_k:
#             dcg = ((ranks<=top_k)*relevance/app).sum(-1)
#             k = min(top_k,scores.shape[-1])
#             sorted_k_relevance = relevance.sort(dim=-1, descending=True).values[...,:k] #get first k items in sorted_relevance on last dimension  
#             idcg = (sorted_k_relevance/torch.log2(torch.arange(1,k+1,device=sorted_k_relevance.device)+1)).sum(-1)
#             ndcg = dcg/idcg # ndcg.shape = (num_samples, lookback)
#             if not self.batch_metric:
#                 setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + ndcg.sum())
#             else:
#                 getattr(self, f"correct@{top_k}").append(ndcg)
#         if not self.batch_metric:
#             self.total += relevance.shape[0]
    
# class MRR(RecMetric):
#     '''
#     Mean Reciprocal Rank (MRR) evaluates the efficacy of a ranking system by considering the placement of the first relevant item within the ranked list.
#     It is calculated by taking the reciprocal of the rank of the first relevant item.
#     It emphasizes that the position of the first relevant item is more important than the placement of the other relevant items.
#     '''
#     def __init__(self, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.batch_metric = batch_metric

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         # Call not_nan_subset to subset scores, relevance where relevance is not nan
#         kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
#         scores, relevance = kwargs["scores"], kwargs["relevance"]

#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1

#         relevant = relevance>0
#         for top_k in self.top_k:
#             mrr = ((ranks<=top_k)*relevant*(1/ranks)).max(-1).values
#             if not self.batch_metric:
#                 setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + mrr.sum())
#             else:
#                 getattr(self, f"correct@{top_k}").append(mrr)
#         if not self.batch_metric:
#             self.total += relevance.shape[0]

# class Precision(RecMetric):
#     '''
#     It computes the proportion of accurately identified relevant items among all the items recommended within a list of length K. 
#     It is used to explicitly count the number of recommended, or retrieved, items that are truly relevant.
#     '''
#     def __init__(self, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.batch_metric = batch_metric

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         """
#         Updates the metric values based on the input scores and relevance tensors.

#         Args:
#             scores (torch.Tensor): Tensor containing prediction scores.
#             relevance (torch.Tensor): Tensor containing relevance values.
#         """
#         # Call not_nan_subset to subset scores, relevance where relevance is not nan
#         kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
#         scores, relevance = kwargs["scores"], kwargs["relevance"]

#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1

#         relevant = relevance>0
#         for top_k in self.top_k:
#             precision = ((ranks<=top_k)*relevant/top_k).sum(-1)
#             if not self.batch_metric:
#                 setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + precision.sum())
#             else:
#                 getattr(self, f"correct@{top_k}").append(precision)
#         if not self.batch_metric:
#             self.total += relevance.shape[0]

# class Recall(RecMetric):
#     '''
#     It assesses the fraction of correctly identified relevant items among the top K recommendations, relative to the total number of relevant items in the dataset. 
#     It measures the effectiveness of the method in capturing relevant items among all of those present in the dataset.
    
#     '''
#     def __init__(self, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.batch_metric=batch_metric

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         # Call not_nan_subset to subset scores, relevance where relevance is not nan
#         kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
#         scores, relevance = kwargs["scores"], kwargs["relevance"]

#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1

#         relevant = relevance>0
#         for top_k in self.top_k:
#             recall = ((ranks<=top_k)*relevant/relevant.sum(-1,keepdim=True)).sum(-1)
#             #torch.minimum(relevant.sum(-1,keepdim=True),top_k*torch.ones_like(relevant.sum(-1,keepdim=True)))
#             if not self.batch_metric:
#                 setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + recall.sum())
#             else:
#                 getattr(self, f"correct@{top_k}").append(recall)
#         if not self.batch_metric:
#             self.total += relevance.shape[0]

# class F1(RecMetric):
#     '''
#     The F1 score is the harmonic mean of precision and recall. 
#     It is a single metric that combines both precision and recall to provide a single measure of the quality of a ranking system.
#     '''
#     def __init__(self, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.precision = Precision(top_k, batch_metric)
#         self.recall = Recall(top_k, batch_metric)

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         self.precision.update(scores, relevance)
#         self.recall.update(scores, relevance)

#     def compute(self):
#         precision = self.precision.compute()
#         recall = self.recall.compute()
#         out = {}
#         for k in self.top_k:
#             out[f"@{k}"] = 2*(precision[f"@{k}"]*recall[f"@{k}"])/(precision[f"@{k}"]+recall[f"@{k}"])
#         return out

# class PrecisionWithRelevance(RecMetric):
#     '''
#     It computes the proportion of accurately identified relevant items among all the items recommended within a list of length K.
#     It is used to explicitly count the number of recommended, or retrieved, items that are truly relevant.
#     '''
#     def __init__(self, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.batch_metric=batch_metric

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         # Call not_nan_subset to subset scores, relevance where relevance is not nan
#         kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
#         scores, relevance = kwargs["scores"], kwargs["relevance"]

#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1

#         for top_k in self.top_k:
#             precision = ((ranks<=top_k)*relevance/(top_k*relevance.sum(-1,keepdim=True))).sum(-1)
#             if not self.batch_metric:
#                 setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + precision.sum())
#             else:
#                 getattr(self, f"correct@{top_k}").append(precision)

#         if not self.batch_metric:
#             self.total = self.total + relevance.shape[0] #not using += cause getting InferenceMode error sometimes

# class MAP(RecMetric):
#     '''
#     Mean Average Precision (MAP) evaluates the efficacy of a ranking system by considering the average precision across the top R recommendations for R ranging from 1 to K. 
#     It emphasizes that precision values for items within the top K positions contribute to the overall assessment also accounting for the significance of the order in the ranking. 
#     Different from NDCG, this metric does not explicitly assign a different importance to different slots.
    
#     '''
#     def __init__(self, top_k=[5,10,20], batch_metric=False):
#         super().__init__(top_k, batch_metric)
#         self.batch_metric=batch_metric

#         self.precision_at_k = PrecisionWithRelevance(list(range(1,torch.max(torch.tensor(self.top_k))+1)), batch_metric)

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         self.precision_at_k.update(scores, relevance)

#     def compute(self):
#         if not self.batch_metric:
#             for top_k in self.top_k:
#                 for k in range(1,top_k+1):
#                     setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + getattr(self.precision_at_k, f"correct@{k}"))
#                 setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}")/top_k)
#             setattr(self,"total", getattr(self.precision_at_k, f"total"))
#         else:
#             for top_k in self.top_k:
#                 correct = getattr(self.precision_at_k, f"correct@{top_k}")
#                 for k in range(1,top_k):
#                     new_correct = getattr(self.precision_at_k, f"correct@{k}")
#                     for i,c in enumerate(new_correct):
#                         correct[i] += c
#                 setattr(self, f"correct@{top_k}", [c/top_k for c in correct])
        
#         return super().compute()
    
#     def reset(self):
#         super().reset()
#         self.precision_at_k.reset()