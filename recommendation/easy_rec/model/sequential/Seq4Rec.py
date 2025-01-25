import torch

class Seq4Rec(torch.nn.Module):
    def __init__(self, 
                 embedding_layer,
                 encoder_layer,
                 output_layer, 
                 **kwargs):
        super().__init__()
        self.reduce_timesteps = ReduceTimesteps()

    def forward(self, input_sequences, output_items):
        embedded_input = self.embedding_layer(input_sequences)

        encoded = self.encoder_layer(embedded_input)

        encoded, output_items = self.reduce_timesteps(encoded, output_items)

        output_scores = self.output_layer(encoded, output_items)

        return output_scores


class ItemEmbedding(torch.nn.Module):
    def __init__(self, num_items, emb_size):
        super().__init__()
        self.item_emb = torch.nn.Embedding(num_items, emb_size)

    def forward(self, input_seqs):
        return self.item_emb(input_seqs)

    
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, lookback, emb_size):
        super().__init__()
        self.pos_emb = torch.nn.Embedding(lookback, emb_size)

    def forward(self, input_seqs):
        positions = torch.tile(torch.arange(input_seqs.shape[1], device=next(self.parameters()).device), [input_seqs.shape[0], 1])
        return self.pos_emb(positions)

    
class ItemPositionalEmbedding(torch.nn.Module):
    def __init__(self, num_items, lookback, emb_size, aggregation='add'):
        super().__init__()
        self.item_emb = ItemEmbedding(num_items, emb_size)
        self.pos_emb = PositionalEmbedding(lookback, emb_size)
        self.aggregation = getattr(torch,aggregation)

    def forward(self, input_seqs):
        return self.aggregation(self.item_emb(input_seqs),self.pos_emb(input_seqs))


class MatrixFactorizationLayer(torch.nn.Module):
    def __init__(self, item_emb):
        super().__init__()
        self.item_emb = item_emb

    def forward(self, encoded, output_items):
        output_items_emb = self.item_emb(output_items)

        # Use only last timesteps in poss_item_seqs --> cut encoded to match output_items
        encoded = encoded[:, -output_items.shape[1]:, :, :] # (B, T, 1, E)

        scores = (encoded * output_items_emb).sum(dim=-1)

        return scores


class LinearLayer(torch.nn.Module):
    def __init__(self, emb_size, num_items, *args,**kwargs):
        super().__init__()
        self.linear_layer = torch.nn.Linear(emb_size, num_items+1, *args, **kwargs)

    def forward(self, encoded, output_items):
        scores = self.linear_layer(encoded)

        scores = scores[:, -output_items.shape[1]:, :]

        scores = torch.gather(scores, -1, output_items) # Get scores for items in poss_item_seqs

        return scores

class ReduceTimesteps(torch.nn.Module):
    def __init__(self, emb_size, num_items, *args,**kwargs):
        super().__init__()
    def forward(self, *args):
        timesteps_to_use = min([x.shape[1] for x in args])

        return [x[:, -timesteps_to_use] for x in args]

# dropout #layer_norm (after embedding, and after encoding)

#class GRU

#class causal attention

#class biderectional attention
#####
        embedded = self.dropout(self.first_layernorm(self.item_emb(input_seqs) + self.pos_emb(positions)))

        attention_mask = ~torch.tril(torch.ones((input_seqs.shape[1], input_seqs.shape[1]), dtype=torch.bool, device=next(self.parameters()).device))

        encoded = self.encoder(embedded, attention_mask)

        log_feats = encoded.unsqueeze(2)

        poss_item_embs = self.item_emb(poss_item_seqs)

        # Use only last timesteps in poss_item_seqs --> cut log_feats to match poss_item_embs
        log_feats = log_feats[:, -poss_item_embs.shape[1]:, :, :] # (B, T, 1, E)

        scores = (log_feats * poss_item_embs).sum(dim=-1)