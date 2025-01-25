import torch

class POP(torch.nn.Module):
    def __init__(self, num_items):
        super(POP, self).__init__()
        self.num_items = num_items

    def forward(self, input_seqs, poss_item_seqs):
        # Count the occurrences of each item in the input_seqs
        item_counts = torch.bincount(input_seqs.flatten(), minlength=self.num_items+1).float()

        # Reverse sort the items based on their popularity
        sorted_items = torch.argsort(item_counts, descending=True)

        # Create scores based on the popularity ranks
        scores = torch.zeros_like(poss_item_seqs, dtype=torch.float32)

        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                item = poss_item_seqs[i, j].item()
                rank = (sorted_items == item).nonzero().item() + 1  # Add 1 because ranks start from 1
                scores[i, j] = 1 / rank  # Inverse of rank as the score

        return scores
