import torch

class SequentialBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        is_not_nan = ~torch.isnan(target)
        return super().forward(input[is_not_nan], target[is_not_nan])
    
# class SequentialCrossEntropyLoss(torch.nn.Module): #torch.nn.CrossEntropyLoss):
#     def __init__(self, eps = 1e-6, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.eps = eps

#     def forward(self, input, target):
#         is_not_nan = ~torch.isnan(target)
#         print("A")
        
#         #Manual computation, cause CrossEntropyLoss returns nan
#         new_target = target
#         new_target[~is_not_nan] = 0
#         print("B")
        
#         exps = torch.exp(input) * is_not_nan
#         exps_sum = exps.sum(dim=-1)
#         print("exps_sum", exps_sum)

#         exps_div = exps/(exps_sum.unsqueeze(-1)+self.eps)
#         exps_div = exps_div * is_not_nan
#         print("exps_div", exps_div)

#         loss = exps_div*torch.log(exps_div)*new_target
#         print("E")

#         loss = -loss.sum(dim=-1)[is_not_nan.any(dim=-1)]
#         print("F")

#         output = loss.mean()
#         print("output", output)

#         # Commented code cause CrossEntropyLoss returns nan
#         # target[is_nan] = 0
#         # input[is_nan] = -100

#         # all_items_nans = is_nan.all(dim=-1)

#         # new_target = target[~all_items_nans]
#         # new_input = input[~all_items_nans]

#         # output = super().forward(input, target)

#         return output

class SequentialBPR(torch.nn.Module):
    def __init__(self, clamp_max=20,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clamp_max = clamp_max

    def forward(self, input, target):
        # Input shape: (batch_size, timesteps, num_items)
        # Output shape: (batch_size, timesteps, num_items)
        is_not_nan = ~torch.isnan(target)

        # Change relevance from 0,1 to -1,1
        new_target = target * 2 - 1
        new_target[~is_not_nan] = 0

        # pair positive and negative items in same timestep
        positive_items = new_target > 0
        negative_items = new_target < 0
        item_pairs = (negative_items.unsqueeze(-1) * positive_items.unsqueeze(-2)).float()
        
        item_per_relevance = input.unsqueeze(-1) - input.unsqueeze(-2)
        item_per_relevance = torch.log(1+torch.exp(torch.clamp(item_per_relevance, max=self.clamp_max)))

        # item_per_relevance has shape (N,T,I,I)
        # item_pairs has shape (N,T,I,I)
        # We want shape (N,T,1). summing on last two dimensions if item_pairs is True
        bpr = torch.einsum('ntij,ntij->nt', item_per_relevance, item_pairs)

        bpr = bpr[is_not_nan.any(dim=-1)].mean()

        return bpr
    
# class SequentialCrossEntropyLoss(torch.nn.Module):
#     def __init__(self, eps = 1e-6, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.eps = eps

#     def forward(self, input, target):
#         # Input shape: (batch_size, timesteps, num_items)
#         # Output shape: (batch_size, timesteps, num_items)
#         is_not_nan = ~torch.isnan(target)

#         new_target = target
#         new_target[~is_not_nan] = 0

#         item_softmax = torch.nn.functional.softmax(input, dim=-1)

#         item_per_relevance = (torch.log(item_softmax)*new_target).sum(-1)
        
#         ce = item_per_relevance[is_not_nan.any(dim=-1)]

#         ce = ce.mean()

#         return ce

class SequentialCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        # Input shape: (batch_size, timesteps, num_items)
        # Output shape: (batch_size, timesteps, num_items)
        is_not_nan = ~torch.isnan(target)

        new_target = target + 0 # Without + 0, new_target will be a view of target and will change target (right?)
        new_target[~is_not_nan] = 0

        new_target = new_target[is_not_nan.any(dim=-1)]
        new_input = input[is_not_nan.any(dim=-1)]

        return super().forward(new_input, new_target)
    
class SequentialGeneralizedBCEWithLogitsLoss(SequentialBCEWithLogitsLoss):
    def __init__(self, beta, eps = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.eps = eps
    
    def forward(self, input, target):
        is_positive = target > 0.5
        new_input, new_target = input+0, target+0 #to force copy?
        
        if self.beta == 0:
            new_input,new_target = new_input[~is_positive], new_target[~is_positive]
        else:
            new_input[is_positive] = self.gamma_transformation(new_input[is_positive])

        return super().forward(new_input, new_target)
    
    def gamma_transformation(self, scores):
        return -torch.log((1+torch.exp(-scores))**self.beta-1+self.eps)
