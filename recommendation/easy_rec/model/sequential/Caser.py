

import torch

class Caser(torch.nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------
    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, lookback, emb_size, num_hor_filters, num_ver_filters, drop_rate, act_conv, act_fc, num_items, num_users, *args, **kwargs):
        super().__init__()

        # activation functions
        self.act_conv = getattr(torch.nn,act_conv)() #activation function for convolution layer (i.e., phi_c in paper)
        self.act_fc = getattr(torch.nn,act_fc)() #activation function for fully-connected layer (i.e., phi_a in paper)

        # user and item embeddings
        self.user_embeddings = torch.nn.Embedding(num_users+1, emb_size)
        self.item_embeddings = torch.nn.Embedding(num_items+1, emb_size)

        # vertical conv layer
        self.conv_v = torch.nn.Conv2d(1, num_ver_filters, (lookback, 1)) #in_channels, out_channels, kernel_size
            
        # horizontal conv layer
        self.conv_h = torch.nn.ModuleList([torch.nn.Conv2d(1, num_hor_filters, (h, emb_size)) for h in range(1,lookback+1)])
        self.num_hor_filters = num_hor_filters

        # fully-connected layer
        self.fc1_dim_v = num_ver_filters * emb_size
        self.fc1_dim_h = num_hor_filters * lookback
        # W1, b1 can be encoded with torch.nn.Linear
        self.fc1 = torch.nn.Linear(self.fc1_dim_v + self.fc1_dim_h, emb_size)
        # W2, b2 are encoded with torch.nn.Embedding, as we don't need to compute scores for all items
        self.W2 = torch.nn.Embedding(num_items+1, emb_size+emb_size)
        self.b2 = torch.nn.Embedding(num_items+1, 1)

        self.dropout = torch.nn.Dropout(drop_rate)

    def forward(self, input_seqs, poss_item_seqs, user_var):
        # Embedding Look-up
        item_embs = self.item_embeddings(input_seqs)
        item_embs = item_embs.unsqueeze(1) #To get channel dimension for convolution
        user_emb = self.user_embeddings(user_var)

        # Vertical conv layer
        out_v = self.conv_v(item_embs) #(N, num_ver_filters, 1, emb_size)
        out_v = out_v.view(-1, self.fc1_dim_v) #(N, num_ver_filters*emb_size)

        # Horizontal conv layer
        out_hs = torch.empty((len(self.conv_h),item_embs.shape[0],self.num_hor_filters,1), device=item_embs.device)
        for i,conv in enumerate(self.conv_h):
            conv_out = self.act_conv(conv(item_embs)) #(N, num_hor_filters, lookback-h+1, 1) #h \in {1,2,...,lookback}
            out_hs[i] = conv_out.max(2).values #(N, num_hor_filters, 1)
        out_h = out_hs.permute(1,0,2,3).reshape(item_embs.shape[0],self.fc1_dim_h) #(N, num_hor_filters*lookback)

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1) #(N, num_ver_filters*emb_size + num_hor_filters*lookback)
        
        out = self.dropout(out) # apply dropout

        # fully-connected layer
        z = self.act_fc(self.fc1(out)) # (N, emb_size)
        x = torch.cat([z, user_emb], -1).unsqueeze(1).unsqueeze(2) # (N, 1, 1, 2*emb_size)

        w2 = self.W2(poss_item_seqs)
        b2 = self.b2(poss_item_seqs)

        x = x[:, -poss_item_seqs.shape[1]:, :, :] # (N, 1, 1, 2*emb_size)

        res = (w2 * x).sum(dim=-1) + b2.squeeze(-1)

        return res




class Caser2(torch.nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------
    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, lookback, emb_size, num_hor_filters, num_ver_filters, drop_rate, act_conv, act_fc, num_items, num_users, *args, **kwargs):
        super().__init__()

        # activation functions
        self.act_conv = getattr(torch.nn,act_conv)() #activation function for convolution layer (i.e., phi_c in paper)
        self.act_fc = getattr(torch.nn,act_fc)() #activation function for fully-connected layer (i.e., phi_a in paper)

        # user and item embeddings
        self.user_embeddings = torch.nn.Embedding(num_users+1, emb_size)
        self.item_embeddings = torch.nn.Embedding(num_items+1, emb_size)

        # vertical conv layer
        self.conv_v = torch.nn.Conv2d(1, num_ver_filters, (lookback, 1)) #in_channels, out_channels, kernel_size
            
        # horizontal conv layer
        self.conv_h = torch.nn.ModuleList([torch.nn.Conv2d(1, num_hor_filters, (h, emb_size)) for h in range(1,lookback+1)])
        self.num_hor_filters = num_hor_filters

        # fully-connected layer
        self.fc1_dim_v = num_ver_filters * emb_size
        self.fc1_dim_h = num_hor_filters * lookback
        # W1, b1 can be encoded with torch.nn.Linear
        self.fc1 = torch.nn.Linear(self.fc1_dim_v + self.fc1_dim_h, emb_size)
        # W2, b2 are encoded with torch.nn.Embedding, as we don't need to compute scores for all items
        self.W2 = torch.nn.Embedding(num_items+1, emb_size+emb_size)
        self.b2 = torch.nn.Embedding(num_items+1, 1)

        self.dropout = torch.nn.Dropout(drop_rate)

    def forward(self, input_seqs, poss_item_seqs, user_var):
        lookback = input_seqs.shape[1]
        # Embedding Look-up
        item_embs = self.item_embeddings(input_seqs)
        item_embs = item_embs.unsqueeze(1) #To get channel dimension for convolution
        user_emb = self.user_embeddings(user_var)

        # Vertical conv layer
        # To keep shape L, we need to pad the input sequence
        pad_item_embs = torch.nn.functional.pad(item_embs, (0, 0, lookback-1, 0)) #(N, 1, L+L-1, emb_size)
        out_v = self.conv_v(pad_item_embs) #(N, num_ver_filters, L, emb_size)
        
        # Reshape (N, num_ver_filters, L, emb_size) to (N, L, num_ver_filters, self.fc1_dim_v)
        out_v = out_v.permute(0,2,1,3).reshape(-1, lookback, self.fc1_dim_v) #(N, L, num_ver_filters*emb_size)

        # Horizontal conv layer
        out_hs = torch.empty((len(self.conv_h),item_embs.shape[0],self.num_hor_filters,lookback,1), device=item_embs.device)
        for h,conv in enumerate(self.conv_h):
            pad_item_embs = torch.nn.functional.pad(item_embs, (0, 0, h, 0)) #(N, 1, L+h, emb_size)
            conv_out = self.act_conv(conv(pad_item_embs)) #(N, num_hor_filters, L, 1) #h \in {1,2,...,lookback}
            out_hs[h] = conv_out.cummax(2).values #(N, num_hor_filters, L, 1)
        # Reshape to (N, L, num_hor_filters*lookback)
        out_h = out_hs.permute(1,3,2,4,0).reshape(item_embs.shape[0],lookback,self.fc1_dim_h)

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 2) #(N, lookback, num_ver_filters*emb_size + num_hor_filters*lookback)
        
        out = self.dropout(out) # apply dropout

        # fully-connected layer
        z = self.act_fc(self.fc1(out)) # (N, lookback, emb_size)
        user_emb = user_emb.unsqueeze(1).repeat(1, lookback, 1) # repeat user emb to have the same shape as z
        x = torch.cat([z, user_emb], -1).unsqueeze(2) # (N, L, 1, 2*emb_size)

        w2 = self.W2(poss_item_seqs)
        b2 = self.b2(poss_item_seqs)

        x = x[:, -poss_item_seqs.shape[1]:, :, :] # (N, L, K, 2*emb_size)

        res = (w2 * x).sum(dim=-1) + b2.squeeze(-1) # (N, L, K)

        return res

