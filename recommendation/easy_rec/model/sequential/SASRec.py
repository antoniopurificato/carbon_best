import torch

class SASRec(torch.nn.Module):

    def __init__(self, 
                 num_items, 
                 lookback, 
                 emb_size, 
                 dropout_rate=0, 
                 num_blocks=1, 
                 num_heads=1, 
                 padding_value=0, 
                 **kwargs):
        '''
        These are the parameters for the SASRec model, defined in the corresponding YAML file.

        Args:
            num_items (int): Number of items in the dataset.
            lookback (int): Number of previous items to consider in the sequence. (length of the sequence)
            emb_size (int): Size of the embedding for items and positions.
            dropout_rate (float): Dropout rate for regularization.
            num_blocks (int): Number of Transformer blocks in the encoder.
            num_heads (int): Number of attention heads in the Transformer model.
            padding_value (int, optional): Padding value for item embeddings. Defaults to 0.
        '''
        super().__init__()

        self.padding_value = padding_value
        
        # Item and position embeddings
        self.item_emb = torch.nn.Embedding(num_items + 1, emb_size, padding_idx=padding_value) #+1 because padding #Another +1 because of the mask token
        self.pos_emb = torch.nn.Embedding(lookback, emb_size)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(emb_size, num_heads, emb_size * 4, dropout_rate,
                                                    torch.nn.GELU(), batch_first = True, norm_first = True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_blocks)
        
        # Layer normalization
        self.last_layernorm = torch.nn.LayerNorm(emb_size, eps=1e-8)

    def forward(self, input_seqs, poss_item_seqs):

        ''' 
        Input:
            input_seqs (torch.Tensor): Tensor containing input item sequences. Shape (batch_size, sequence_length).
            poss_item_seqs (torch.Tensor): Tensor containing possible item sequences. Shape (batch_size, output_seq_len, num_items)

        Output:
            scores (torch.Tensor): Tensor containing interaction scores between input and possible items. Shape (batch_size, output_seq_len, num_items)
        '''

        positions = torch.tile(torch.arange(input_seqs.shape[1],
                                            device=next(self.parameters()).device),
                                            [input_seqs.shape[0], 1])

        embedded = self.dropout(self.item_emb(input_seqs) + self.pos_emb(positions))

        attention_mask = ~torch.tril(torch.ones((input_seqs.shape[1], input_seqs.shape[1]),
                                                dtype=torch.bool,
                                                device=next(self.parameters()).device))

        encoded = self.encoder(embedded, attention_mask)

        log_feats = self.last_layernorm(encoded).unsqueeze(2)

        poss_item_embs = self.item_emb(poss_item_seqs)

        # Use only last timesteps in poss_item_seqs --> cut log_feats to match poss_item_embs
        log_feats = log_feats[:, -poss_item_embs.shape[1]:, :, :] # (B, T, 1, E)

        scores = (log_feats * poss_item_embs).sum(dim=-1)

        return scores
    


class SASRec2(torch.nn.Module):

    def __init__(self, 
                 num_items, 
                 lookback, 
                 emb_size, 
                 dropout_rate, 
                 num_blocks, 
                 num_heads, 
                 padding_value=0, 
                 **kwargs):
        '''
    These are the parameters for the SASRec model, defined in the corresponding YAML file.

    Args:
        num_items (int): Number of items in the dataset.
        lookback (int): Number of previous items to consider in the sequence. (length of the sequence)
        emb_size (int): Size of the embedding for items and positions.
        dropout_rate (float): Dropout rate for regularization.
        num_blocks (int): Number of Transformer blocks in the encoder.
        num_heads (int): Number of attention heads in the Transformer model.
        padding_value (int, optional): Padding value for item embeddings. Defaults to 0.
        '''
        super().__init__()

        self.padding_value = padding_value
        
        # Item and position embeddings
        self.item_emb = torch.nn.Embedding(num_items + 1, emb_size, padding_idx=padding_value) #+1 because padding #Another +1 because of the mask token
        self.pos_emb = torch.nn.Embedding(lookback, emb_size)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Layer normalization
        self.first_layernorm = torch.nn.LayerNorm(emb_size, eps=1e-8)

        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(emb_size, num_heads, emb_size * 4, dropout_rate,
                                                    torch.nn.GELU(), batch_first = True, norm_first = True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_blocks)
        
        # Layer normalization
        self.last_layernorm = torch.nn.LayerNorm(emb_size, eps=1e-8)

    def forward(self, input_seqs, poss_item_seqs):

        ''' 
    Input:
        input_seqs (torch.Tensor): Tensor containing input item sequences. Shape (batch_size, sequence_length).
        poss_item_seqs (torch.Tensor): Tensor containing possible item sequences. Shape (batch_size, input_seq_len, output_seq_len, num_items)

    Output:
        scores (torch.Tensor): Tensor containing interaction scores between input and possible items. Shape (batch_size, input_seq_len, output_seq_len, num_items)

        '''

        positions = torch.tile(torch.arange(input_seqs.shape[1], device=next(self.parameters()).device), [input_seqs.shape[0], 1])

        embedded = self.dropout(self.first_layernorm(self.item_emb(input_seqs) + self.pos_emb(positions)))

        attention_mask = ~torch.tril(torch.ones((input_seqs.shape[1], input_seqs.shape[1]), dtype=torch.bool, device=next(self.parameters()).device))

        encoded = self.encoder(embedded, attention_mask)

        log_feats = encoded.unsqueeze(2)

        poss_item_embs = self.item_emb(poss_item_seqs)

        # Use only last timesteps in poss_item_seqs --> cut log_feats to match poss_item_embs
        log_feats = log_feats[:, -poss_item_embs.shape[1]:, :, :] # (B, T, 1, E)

        scores = (log_feats * poss_item_embs).sum(dim=-1)

        return scores