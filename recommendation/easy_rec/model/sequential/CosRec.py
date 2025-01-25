import torch

class CosRec(torch.nn.Module):
    '''
    A 2D CNN for sequential Recommendation.

    Args:
        num_users: number of users.
        num_items: number of items.
        seq_len: length of sequence, Markov order.
        embed_dim: dimensions for user and item embeddings.
        block_num: number of cnn blocks.
        block_dim: the dimensions for each block. len(block_dim)==block_num
        fc_dim: dimension of the first fc layer, mainly for dimension reduction after CNN.
        ac_fc: type of activation functions.
        drop_prob: dropout ratio.
    '''
    def __init__(self, num_users, num_items, emb_size, block_dims, fc_dim, act_fc, dropout_rate=0.5, **kwargs):
        super().__init__()

        # user and item embeddings
        self.user_embeddings = torch.nn.Embedding(num_users+1, emb_size)
        self.in_item_embeddings = torch.nn.Embedding(num_items+1, emb_size)

        ### build cnnBlock
        block_dims = block_dims.copy()
        block_dims.insert(0, 2*emb_size) #add input dimension
        cnn_blocks = []
        for i in range(len(block_dims)-1):
            cnn_blocks.append(CnnBlock(block_dims[i], block_dims[i+1]))
        self.cnn_blocks = torch.nn.ModuleList(cnn_blocks)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        ### dropout and fc layer
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc1 = torch.nn.Linear(block_dims[-1], fc_dim)
        self.act_fc = getattr(torch.nn,act_fc)() #activation function for fully-connected layer (i.e., phi_a in paper)

        ### Output layer
        self.fc_out = torch.nn.Linear(emb_size+fc_dim, num_items+1)

    def forward(self, item_seq, items_to_predict, user_ids):
        '''
        #TODO: check dimensions
        Args:
            seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
                a batch of sequence
            user_var: torch.LongTensor with size [batch_size]
                a batch of user
            item_var: torch.LongTensor with size [batch_size]
                a batch of items
            for_pred: boolean, optional
                Train or Prediction. Set to True when evaluation.
        '''
        item_embs = self.in_item_embeddings(item_seq) # (b, L, embed)
        user_emb = self.user_embeddings(user_ids) # (b, 1, embed)

        # cast all item embeddings pairs against each other
        item_i = item_embs.unsqueeze(1).repeat(1, item_embs.shape[1], 1, 1) # (b, 1, L, embed) --> (b, L, L, embed)
        item_j = item_embs.unsqueeze(2).repeat(1, 1, item_embs.shape[1], 1) # (b, L, 1, embed) --> (b, L, L, embed)
        all_embed = torch.cat([item_i, item_j], 3) # (b, L, L, 2*embed)
        out = all_embed.permute(0, 3, 1, 2) # (b, 2*embed, L, L) # Move embedding dimension to the front (channel first for CNN)

        # 2D CNN
        for cnn in self.cnn_blocks:
            out = cnn(out)
        out = self.avgpool(out).squeeze(-1).squeeze(-1) # (b, block_dims[-1])

        # apply fc and dropout
        out = self.dropout(self.act_fc(self.fc1(out)))
        
        # # add user embedding everywhere
        # usr = user_emb.repeat(1, item_seq.shape[1], 1) # (b, 5, embed)
        # usr = torch.unsqueeze(usr, 2) # (b, 5, 1, embed)
        x = torch.cat([out, user_emb.squeeze(1)], -1)

        scores = self.fc_out(x).unsqueeze(1)

        timesteps_to_use = min(items_to_predict.shape[1], scores.shape[1])

        scores = scores[:, -timesteps_to_use:]
        items_to_predict = items_to_predict[:, -timesteps_to_use:]

        scores = torch.gather(scores, -1, items_to_predict) # Get scores for items in items_to_predict

        return scores

class CnnBlock(torch.nn.Module):
    '''
    CNN block with two layers.
    
    For this illustrative model, we don't change stride and padding. 
    But to design deeper models, some modifications might be needed.
    '''
    def __init__(self, input_dim, output_dim, stride=1, padding=0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride, padding=padding)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(output_dim)
        self.bn2 = torch.nn.BatchNorm2d(output_dim)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out
    



class CosRec2(torch.nn.Module):
    '''
    A 2D CNN for sequential Recommendation.

    Args:
        num_users: number of users.
        num_items: number of items.
        seq_len: length of sequence, Markov order.
        embed_dim: dimensions for user and item embeddings.
        block_num: number of cnn blocks.
        block_dim: the dimensions for each block. len(block_dim)==block_num
        fc_dim: dimension of the first fc layer, mainly for dimension reduction after CNN.
        ac_fc: type of activation functions.
        drop_prob: dropout ratio.
    '''
    def __init__(self, num_users, num_items, emb_size, block_dims, fc_dim, act_fc, dropout_rate=0.5, **kwargs):
        super().__init__()

        # user and item embeddings
        self.user_embeddings = torch.nn.Embedding(num_users+1, emb_size)
        self.in_item_embeddings = torch.nn.Embedding(num_items+1, emb_size)

        ### build cnnBlock
        block_dims = block_dims.copy()
        block_dims.insert(0, 2*emb_size) #add input dimension
        cnn_blocks = []
        for i in range(len(block_dims)-1):
            cnn_blocks.append(CnnBlock2(block_dims[i], block_dims[i+1]))
        self.cnn_blocks = torch.nn.ModuleList(cnn_blocks)
        self.avgpool = SpecialPooling()

        ### dropout and fc layer
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc1 = torch.nn.Linear(block_dims[-1], fc_dim)
        self.act_fc = getattr(torch.nn,act_fc)() #activation function for fully-connected layer (i.e., phi_a in paper)

        ### Output layer
        self.fc_out = torch.nn.Linear(emb_size+fc_dim, num_items+1)

    def forward(self, item_seq, items_to_predict, user_ids):
        '''
        #TODO: check dimensions
        Args:
            seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
                a batch of sequence
            user_var: torch.LongTensor with size [batch_size]
                a batch of user
            item_var: torch.LongTensor with size [batch_size]
                a batch of items
            for_pred: boolean, optional
                Train or Prediction. Set to True when evaluation.
        '''
        item_embs = self.in_item_embeddings(item_seq) # (b, L, embed)
        user_emb = self.user_embeddings(user_ids) # (b, 1, embed)

        # cast all item embeddings pairs against each other
        item_i = item_embs.unsqueeze(1).repeat(1, item_embs.shape[1], 1, 1) # (b, 1, L, embed) --> (b, L, L, embed)
        item_j = item_embs.unsqueeze(2).repeat(1, 1, item_embs.shape[1], 1) # (b, L, 1, embed) --> (b, L, L, embed)
        all_embed = torch.cat([item_i, item_j], 3) # (b, L, L, 2*embed)
        out = all_embed.permute(0, 3, 1, 2) # (b, 2*embed, L, L) # Move embedding dimension to the front (channel first for CNN)

        # 2D CNN
        for cnn in self.cnn_blocks:
            out = cnn(out)
        out = self.avgpool(out).squeeze(-1).squeeze(-1) # (b, block_dims[-1])

        # apply fc and dropout
        out = self.dropout(self.act_fc(self.fc1(out)))
        
        # # add user embedding everywhere
        user_emb = user_emb.unsqueeze(1).repeat(1, out.shape[1], 1) # repeat user emb to have the same shape as 
        x = torch.cat([out, user_emb.squeeze(1)], -1)

        scores = self.fc_out(x)

        timesteps_to_use = min(items_to_predict.shape[1], scores.shape[1])

        scores = scores[:, -timesteps_to_use:]
        items_to_predict = items_to_predict[:, -timesteps_to_use:]

        scores = torch.gather(scores, -1, items_to_predict) # Get scores for items in items_to_predict

        return scores

class CnnBlock2(torch.nn.Module):
    '''
    CNN block with two layers.
    
    For this illustrative model, we don't change stride and padding. 
    But to design deeper models, some modifications might be needed.
    '''
    def __init__(self, input_dim, output_dim, stride=1, padding=0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride, padding=padding)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(output_dim)
        self.bn2 = torch.nn.BatchNorm2d(output_dim)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        # to keep the same shape, we need to add padding
        # we need to left pad the last two dimensions (height and width)
        # using the kernel size of the next layer
        out = torch.nn.functional.pad(out, (self.conv2.kernel_size[-1]-1, 0, self.conv2.kernel_size[-2]-1, 0))
        out = self.relu(self.bn2(self.conv2(out)))
        return out
    
class SpecialPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x shape (B, e, L, L)
        L = x.shape[-1]
        mask = self.create_mask(L, x.device)
        out = torch.einsum("nejk,tjk->nte", x, mask)/(torch.arange(1,L+1,device=x.device).float()[:,None])**2
        return out

    def create_mask(self, L, device="cpu"):
        x = torch.triu(torch.ones(L, L, dtype=int, device=device))
        x = x[None,:]*x.T[:,None]
        x = x + x.transpose(1,2)
        x = 1*(x>0).float()
        return x