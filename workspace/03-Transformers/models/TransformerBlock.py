import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
        Self Attention Class Module for Transformer
    """
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        assert (self.heads_dim * heads == embed_size) # Embed size needs to b divisible by the number of heads

        # Following the Attention mechanism formula from the paper
        # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        # d_k = embed_size // heads
        # Q = queries, K = keys, V = values
        # fc_out = fully connected layer to convert the concatenated heads back to the original embed_size
        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.heads_dim, embed_size)


    def forward(self, values, keys, query, mask):
        """
            Method to perform one forward pass through a batch of the self attention layer
            Inputs:
                values {tensor}: values of the input
                keys {tensor}: keys of the input
                query {tensor}: query of the input
                mask {tensor}: mask of the input
            Outputs:
                out {tensor}: output of the self attention layer
        """

        # define the number of sample
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]


        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries = query.reshape(N, query_len, self.heads, self.heads_dim)

        # Force the dimensions to be compatible for matrix multiplication
        values = self.values(values) # (N, value_len, heads, heads_dim)
        keys = self.keys(keys) # (N, key_len, heads, heads_dim)
        queries = self.queries(queries) # (N, query_len, heads, heads_dim)

        # Compute the dot product between the query and the keys
        # for each word in the input sentence, how much do we need to pay attention to each word in the input sentence
        # the higher the dot product, the more attention we pay to that word
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)


        # Apply the mask
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        # Normalize the attention scores across the source sentence
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)


        # compute the final output of the self attention layer
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.heads_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out shape: (N, query_len, heads, heads_dim) then flatten last two dimensions by reshaping

        out = self.fc_out(out)
        return out
    

class TransformerBlock(nn.Module):
    """
        Class Module to Construct the Transformer Block
    """
    def __init__(self, embed_size, heads, dropout, forward_expansion) :
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.FeedForward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, values, keys, query, mask):
        """
            Method to perform one forward pass through a batch of the transformer block
            Inputs:
                values {tensor}: values of the input
                keys {tensor}: keys of the input
                query {tensor}: query of the input
                mask {tensor}: mask of the input
            Outputs:
                out {tensor}: output of the self attention layer
        """

        # start the attention layer
        attention = self.attention(values, keys, query, mask)
        
        # first skip connection
        x = self.dropout(self.norm1(attention + query))

        # pass through the feed forward layer
        self.forward = self.FeedForward(x)

        # add and normalization
        out = self.dropout(self.norm2(self.forward + x))
        return out