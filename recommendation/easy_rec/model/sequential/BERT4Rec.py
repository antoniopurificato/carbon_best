import torch

class BERT4Rec(torch.nn.Module):
    def __init__(self, 
                 num_items, 
                 emb_size, 
                 lookback, 
                 bert_num_blocks=1, 
                 bert_num_heads=1, 
                 dropout_rate=0, 
                 padding_value=0, 
                 **kwargs):
        '''
    args:
        num_items (int): Number of items in the dataset.
        emb_size (int): Size of the item and position embeddings.
        lookback (int): Number of previous items to consider in the sequence. (length of the sequence)
        bert_num_blocks (int): Number of Transformer blocks in the encoder.
        bert_num_heads (int): Number of attention heads in the Transformer model.
        dropout_rate (float): Dropout rate for regularization.
        padding_value (int, optional): Padding value for item embeddings. Defaults to 0.
    '''
        super().__init__()

        self.padding_value = padding_value

        self.item_emb = torch.nn.Embedding(num_items + 2, emb_size, padding_idx=padding_value) #+1 because padding #Another +1 because of the mask token
        self.pos_emb = torch.nn.Embedding(lookback, emb_size)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        #TODO: make feedforward dim a parameter; also activation

        encoder_layer = torch.nn.TransformerEncoderLayer(emb_size, bert_num_heads, emb_size * 4, dropout_rate,
                                                    torch.nn.GELU(), batch_first = True, norm_first = True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, bert_num_blocks)

        self.out = torch.nn.Linear(emb_size, num_items + 2)

    def forward(self, input_seqs, poss_item_seqs):
        ''' 
    Input:
        input_seqs (torch.Tensor): Tensor containing input item sequences. Shape (batch_size, sequence_length).
        poss_item_seqs (torch.Tensor): Tensor containing possible item sequences. Shape (batch_size, input_seq_len, output_seq_len, num_items)

    Output:
        scores (torch.Tensor): Tensor containing interaction scores between input and possible items. Shape (batch_size, input_seq_len, output_seq_len, num_items)

        '''
        # Create mask to exclude padding values from attention
        mask = torch.isclose(input_seqs, self.padding_value*torch.ones_like(input_seqs)).unsqueeze(1).repeat(
            self.encoder.layers[0].self_attn.num_heads, input_seqs.shape[1], 1)

        # Generate positions tensor
        positions = torch.tile(torch.arange(input_seqs.shape[1],
                                            device=next(self.parameters()).device),
                                            [input_seqs.shape[0], 1])

        embedded = self.dropout(self.item_emb(input_seqs) + self.pos_emb(positions))

        encoded = self.encoder(embedded, mask)

        scores = self.out(encoded)

        scores = scores[:, -poss_item_seqs.shape[1]:, :]

        scores = torch.gather(scores, -1, poss_item_seqs) # Get scores for items in poss_item_seqs

        return scores