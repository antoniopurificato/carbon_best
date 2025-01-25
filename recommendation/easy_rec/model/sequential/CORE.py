import torch

class CORE(torch.nn.Module):
    def __init__(self, num_items, emb_size, sess_dropout_rate, item_dropout_rate, padding_value=0, **kwargs):
        super().__init__()
        
        self.padding_value = padding_value

        self.sess_dropout = torch.nn.Dropout(sess_dropout_rate)
        self.item_dropout = torch.nn.Dropout(item_dropout_rate)

        self.item_embedding = torch.nn.Embedding(num_items+1, emb_size, padding_idx=padding_value) #+1 for padding
    
    def compute_alpha_avg(self, item_seq):
        mask = ~torch.isclose(item_seq, self.padding_value*torch.ones_like(item_seq))
        alpha = mask/mask.sum(dim=1, keepdim=True)
        return alpha
    
    def compute_alpha_transformer(self, item_seq):
        pass

    def forward(self, input_seqs, items_to_predict):
        item_embs = self.item_embedding(input_seqs)
        x = self.sess_dropout(item_embs)
        
        alpha = self.compute_alpha_avg(input_seqs).unsqueeze(-1)

        seq_output = (alpha * x).sum(dim=1).unsqueeze(1)

        seq_output = torch.nn.functional.normalize(seq_output, dim=-1)

        timesteps_to_use = min(items_to_predict.shape[1], seq_output.shape[1])

        seq_output = seq_output[:, -timesteps_to_use:].unsqueeze(2)
        items_to_predict = items_to_predict[:, -timesteps_to_use:]

        poss_item_embs = torch.nn.functional.normalize(self.item_dropout(self.item_embedding(items_to_predict)), dim=-1)

        scores = (seq_output * poss_item_embs).sum(dim=-1)

        return scores