import torch

class GRU4Rec(torch.nn.Module):
    def __init__(self, 
                 num_items,
                 emb_size, 
                 hidden_size, 
                 num_layers=1,
                 dropout_hidden=0, 
                 dropout_input=0,
                 padding_value = 0,
                 **kwargs):
        '''
    args:
        num_items (int): Number of items in the dataset.
        hidden_size (int): Size of the hidden state in the GRU.
        num_layers (int, optional): Number of layers in the GRU. Defaults to 1.
        dropout_hidden (float, optional): Dropout rate for hidden states in the GRU. Defaults to 0.
        dropout_input (float, optional): Dropout rate for input embeddings. Defaults to 0.
        emb_size (int, optional): Size of the item embedding. Defaults to 50.
    '''
        
        super(GRU4Rec, self).__init__()

        # Initialize model parameters
        self.num_items = num_items
        
        hidden = torch.zeros(num_layers, hidden_size, requires_grad=True)
        self.register_buffer("hidden", hidden) #register buffer is needed to move the tensor to the right device

        # Dropout layer for input embeddings
        self.inp_dropout = torch.nn.Dropout(p=dropout_input)

        # Linear layer for output logits
        self.h2o = torch.nn.Linear(hidden_size, num_items+1)

        # Item embedding layer
        self.look_up = torch.nn.Embedding(num_items+1, emb_size, padding_idx=padding_value)
        
        # GRU layer
        self.gru = torch.nn.GRU(emb_size, hidden_size, num_layers, dropout=dropout_hidden, batch_first=True)

    def forward(self, input_seqs, poss_item_seqs):

        ''' 
    Input:
        input_seqs (torch.Tensor): Tensor containing input item sequences. Shape (batch_size, sequence_length).
        poss_item_seqs (torch.Tensor): Tensor containing possible item sequences. Shape (batch_size, input_seq_len, output_seq_len, num_items)

    Output:
        scores (torch.Tensor): Tensor containing interaction scores between input and possible items. Shape (batch_size, input_seq_len, output_seq_len, num_items)

        '''

        embedded = self.look_up(input_seqs)

        embedded = self.inp_dropout(embedded)

        output, hidden = self.gru(embedded, torch.tile(self.hidden, (1, input_seqs.shape[0], 1)))

        scores = self.h2o(output)

        scores = scores[:, -poss_item_seqs.shape[1]:, :]

        scores = torch.gather(scores, -1, poss_item_seqs) # Get scores for items in poss_item_seqs

        return scores