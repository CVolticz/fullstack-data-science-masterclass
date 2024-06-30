import torch
import torch.nn as nn
from TransformerBlock import SelfAttention, TransformerBlock


class DecoderBlock(nn.Module):
    """
        Class Module to Construct the Decoder Block
    """
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.device = device
        self.norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, value, key, src_mask, trg_mask):
        """
            Method to perform one forward pass through a batch of the decoder block
            Inputs:
                x {tensor}: input tensor
                value {tensor}: value tensor
                key {tensor}: key tensor
                src_mask {tensor}: source mask tensor to avoid attending to the padding tokens
                trg_mask {tensor}: target mask tensor essential to avoid the decoder to look ahead
            Outputs:
                out {tensor}: output of the decoder block
        """

        # start the attention layer
        attention = self.attention(x, x, x, trg_mask)

        # first skip connection
        query = self.dropout(self.norm(attention + x))

        # start the transformer block
        out = self.transformer_block(value, key, query, src_mask)

        return out
    

class Decoder(nn.Module):
    """
        Class Module to Construct the Decoder Block
    """
    def __init__(self, 
        target_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # expand the transformer block for number of layers specified
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    device=device
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_out, src_mask, trg_mask):
        """
            Method to perform one forward pass through a batch of the decoder
            Inputs:
                x {tensor}: input tensor
                enc_out {tensor}: encoder output tensor
                src_mask {tensor}: source mask tensor to avoid attending to the padding tokens
                trg_mask {tensor}: target mask tensor essential to avoid the decoder to look ahead
            Outputs:
                out {tensor}: output of the decoder block
        """

        # define the number of sample and sequence length
        N, seq_length = x.shape

        # instantiate the position tensor with a tensor of 0 for every example across the sequence length
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # embed the word and position tensors
        x = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )


        # iterate through the transformer blocks
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)


        # compute the final output of the decoder
        out = self.fc_out(x)
        return out