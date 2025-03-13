import torch
import torch.nn as nn
import torch.nn.functional as F


class InformerEncoder(nn.Module):
    """
    Informer Encoder module that stacks multiple encoder layers.

    Args:
        encoder_layer (nn.Module): The encoder layer to be used.
        num_layers (int): The number of encoder layers to stack.

    Attributes:
        layers (nn.ModuleList): A list of encoder layers.
    """

    def __init__(self, encoder_layer, num_layers):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass of the Informer Encoder.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            src_key_padding_mask (Tensor, optional): Mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor after passing through all encoder layers.
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


class InformerEncoderLayer(nn.Module):
    """
    Informer Encoder Layer module.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float, optional): The dropout value. Default: 0.1.

    Attributes:
        attn (ProbSparseAttention): The attention layer.
        feedforward (nn.Sequential): The feedforward layer.
        layernorm1 (nn.LayerNorm): Layer normalization after attention.
        layernorm2 (nn.LayerNorm): Layer normalization after feedforward.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(InformerEncoderLayer, self).__init__()
        self.attn = ProbSparseAttention(d_model, nhead, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass of the Informer Encoder Layer.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            src_key_padding_mask (Tensor, optional): Mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor after attention and feedforward layers.
        """
        attn_out = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.layernorm1(x + self.dropout(attn_out))
        ff_out = self.feedforward(x)
        x = self.layernorm2(x + self.dropout(ff_out))
        return x


class ProbSparseAttention(nn.Module):
    """
    Probabilistic Sparse Attention module.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dropout (float, optional): The dropout value. Default: 0.1.

    Attributes:
        attn (nn.MultiheadAttention): The multi-head attention layer.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Forward pass of the Probabilistic Sparse Attention.

        Args:
            query (Tensor): The query tensor.
            key (Tensor): The key tensor.
            value (Tensor): The value tensor.
            key_padding_mask (Tensor, optional): Mask for the keys per batch (optional).

        Returns:
            Tensor: The output after attention mechanism.
        """
        attn_output, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        return attn_output


class PerformerEncoder(nn.Module):
    """
    Performer Encoder module that stacks multiple encoder layers.

    Args:
        encoder_layer (nn.Module): The encoder layer to be used.
        num_layers (int): The number of encoder layers to stack.

    Attributes:
        layers (nn.ModuleList): A list of encoder layers.
    """

    def __init__(self, encoder_layer, num_layers):
        super(PerformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass of the Performer Encoder.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            src_key_padding_mask (Tensor, optional): Mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor after passing through all encoder layers.
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


class PerformerEncoderLayer(nn.Module):
    """
    Performer Encoder Layer module.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float, optional): The dropout value. Default: 0.1.

    Attributes:
        attn (LinearAttention): The linear attention layer.
        feedforward (nn.Sequential): The feedforward layer.
        layernorm1 (nn.LayerNorm): Layer normalization after attention.
        layernorm2 (nn.LayerNorm): Layer normalization after feedforward.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(PerformerEncoderLayer, self).__init__()
        self.attn = LinearAttention(d_model, nhead, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass of the Performer Encoder Layer.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            src_key_padding_mask (Tensor, optional): Mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor after attention and feedforward layers.
        """
        attn_out = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.layernorm1(x + self.dropout(attn_out))
        ff_out = self.feedforward(x)
        x = self.layernorm2(x + self.dropout(ff_out))
        return x


class LinearAttention(nn.Module):
    """
    Linear Attention module for Performer.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dropout (float, optional): The dropout value. Default: 0.1.

    Attributes:
        nhead (int): The number of attention heads.
        d_k (int): The dimension of keys/queries in each head.
        w_q (nn.Linear): Linear transformation for queries.
        w_k (nn.Linear): Linear transformation for keys.
        w_v (nn.Linear): Linear transformation for values.
        out_proj (nn.Linear): Linear transformation for output.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super(LinearAttention, self).__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Forward pass of the Linear Attention.

        Args:
            query (Tensor): The query tensor of shape (seq_len, batch_size, d_model).
            key (Tensor): The key tensor of shape (seq_len, batch_size, d_model).
            value (Tensor): The value tensor of shape (seq_len, batch_size, d_model).
            key_padding_mask (Tensor, optional): Mask for the keys per batch (optional).

        Returns:
            Tensor: The output after linear attention mechanism.
        """
        seq_len, bsz, _ = query.size()

        # Linear transformations
        q = self.w_q(query).view(seq_len, bsz * self.nhead, self.d_k).transpose(0, 1)
        k = self.w_k(key).view(seq_len, bsz * self.nhead, self.d_k).transpose(0, 1)
        v = self.w_v(value).view(seq_len, bsz * self.nhead, self.d_k).transpose(0, 1)

        # Apply ELU + 1
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            k = k.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), 0)

        # Linear attention
        kv = torch.bmm(k.transpose(1, 2), v)
        qkv = torch.bmm(q, kv)

        # Reshape and apply output projection
        attn_output = qkv.transpose(0, 1).contiguous().view(seq_len, bsz, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output
