import torch

class NARM(torch.nn.Module):
    def __init__(self, 
                 num_items, 
                 hidden_size, 
                 emb_size, 
                 n_layers = 1, 
                 emb_dropout = 0.25, 
                 ct_dropout = 0.5, 
                 **kwargs):
        
        '''
    Args:
        num_items (int): Number of items in the dataset.
        hidden_size (int): Size of the hidden state in the GRU.
        emb_size (int): Size of the item embedding.
        n_layers (int, optional): Number of layers in the GRU. Defaults to 1.
        emb_dropout (float, optional): Dropout rate for item embeddings. Defaults to 0.25.
        ct_dropout (float, optional): Dropout rate for the context vector. Defaults to 0.5.
    '''

        super(NARM, self).__init__()

        # Initialize model parameters
        self.n_items = num_items
        self.hidden_size = hidden_size

        self.n_layers = n_layers

        hidden = torch.zeros((self.n_layers, self.hidden_size), requires_grad=True)
        self.register_buffer('hidden', hidden)

         # Item embedding layer
        self.emb = torch.nn.Embedding(self.n_items+1, emb_size, padding_idx = 0)
        self.emb_dropout = torch.nn.Dropout(emb_dropout)

        # GRU layer
        self.gru = torch.nn.GRU(emb_size, self.hidden_size, self.n_layers, batch_first=True)

         # Attention mechanism parameters
        self.a_1 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = torch.nn.Linear(self.hidden_size, 1, bias=False)

        # Context vector parameters
        self.ct_dropout = torch.nn.Dropout(ct_dropout)
        self.b = torch.nn.Linear(emb_size, 2 * self.hidden_size, bias=False)


    def forward(self, input_seqs, poss_item_seqs):
        
        ''' 
    Input:
        input_seqs (torch.Tensor): Tensor containing input item sequences. Shape (batch_size, sequence_length).
        poss_item_seqs (torch.Tensor): Tensor containing possible item sequences. Shape (batch_size, input_seq_len, output_seq_len, num_items)

    Output:
        scores (torch.Tensor): Tensor containing interaction scores between input and possible items. Shape (batch_size, input_seq_len, output_seq_len, num_items)

        '''
        embs = self.emb_dropout(self.emb(input_seqs))

        gru_out, last_hidden = self.gru(embs, torch.tile(self.hidden, (1, input_seqs.shape[0], 1)))

        c_global = last_hidden[-1]
        
        # Attention mechanism computations
        q1 = self.a_1(gru_out)
        q2 = self.a_2(last_hidden).permute(1, 0, 2)

        alpha = self.v_t(torch.sigmoid(q1 + q2)) #sort of attention
        c_local = torch.sum(alpha * gru_out, 1)
        c_t = torch.cat([c_local, c_global], -1)
        c_t = self.ct_dropout(c_t).unsqueeze(1).unsqueeze(1)
        
         # Embedding possible item sequences
        poss_item_embs = self.b(self.emb(poss_item_seqs))

        timesteps_to_use = min(poss_item_seqs.shape[1], c_t.shape[1])
        c_t = c_t[:,-timesteps_to_use:,:]
        poss_item_embs = poss_item_embs[:,-timesteps_to_use:,:]
        
        scores = (c_t * poss_item_embs).sum(dim=-1)
        return scores   