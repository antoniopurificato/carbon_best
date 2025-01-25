import torch

class HGN(torch.nn.Module):
    def __init__(self, lookback, emb_size, num_items, num_users, **kwargs):
        super(HGN, self).__init__()

        self.user_embeddings = torch.nn.Embedding(num_users+1, emb_size) #+1 because of padding
        self.in_item_embeddings = torch.nn.Embedding(num_items+1, emb_size, padding_idx=0) #+1 because of padding

        self.feature_gate_item = torch.nn.Linear(emb_size, emb_size, bias=False)
        self.feature_gate_user = torch.nn.Linear(emb_size, emb_size)

        self.instance_gate_item = torch.nn.Linear(emb_size, 1, bias=False)
        self.instance_gate_user = torch.nn.Linear(emb_size, lookback, bias=False)

        self.out_item_embeddings = torch.nn.Embedding(num_items+1, emb_size, padding_idx=0) #+1 because of padding

    def forward(self, item_seq, items_to_predict, user_ids):
        in_item_embs = self.in_item_embeddings(item_seq)
        user_emb = self.user_embeddings(user_ids)

        # feature gating
        gated_item = in_item_embs * torch.sigmoid(self.feature_gate_item(in_item_embs) + self.feature_gate_user(user_emb).unsqueeze(1))

        # instance gating
        union_out = gated_item * torch.sigmoid(self.instance_gate_item(gated_item) + self.instance_gate_user(user_emb).unsqueeze(-1))
        #union_out = union_out.cumsum(dim=1) / torch.arange(1, union_out.shape[1]+1, device=union_out.device).unsqueeze(0).unsqueeze(-1) #average over time
        union_out = union_out.mean(dim=1, keepdim=True)

        out_item_embs = self.out_item_embeddings(items_to_predict) #torch.Size([B, T, P, H])

        user_emb = user_emb.unsqueeze(1).unsqueeze(2) #---> torch.Size([B, 1, 1, H])
        union_out = union_out.unsqueeze(2) #---> torch.Size([B, L, 1, H])
        in_item_embs = in_item_embs.unsqueeze(2) #---> torch.Size([B, L, 1, H])
        
        timesteps_to_use = min(items_to_predict.shape[1], union_out.shape[1])

        union_out = union_out[:, -timesteps_to_use:]
        out_item_embs = out_item_embs[:, -timesteps_to_use:]

        # user-level
        scores = (out_item_embs * user_emb).sum(dim=-1)

        # union-level
        scores += (out_item_embs * union_out).sum(dim=-1)

        # item-item product
        # attn = torch.tril(torch.ones(in_item_embs.shape[1],in_item_embs.shape[1], device=out_item_embs.device))[-out_item_embs.shape[1]:].unsqueeze(0).unsqueeze(-1)
        # res += ((out_item_embs.unsqueeze(2) * in_item_embs.unsqueeze(1)).sum(dim=-1)*attn).sum(dim=2) / attn.sum(2) #average over time
        
        # item-item product
        scores += (out_item_embs.unsqueeze(2) * in_item_embs.unsqueeze(1)).sum(dim=-1).sum(dim=2)
        
        return scores