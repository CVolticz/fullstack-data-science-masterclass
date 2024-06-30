import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock


class Encoder(nn.Module):
    """
        Class Module to Construct the Encoder Block
    """
    def __init__(self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # expand the transformer block for number of layers specified
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask):
        """
            Method to perform one forward pass through a batch of the encoder
        """

        # define the number of sample and sequence length
        N, seq_length = x.shape

        # instantiate the position tensor with a tensor of 0 for every example across the sequence length
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # embed the word and position tensors
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )

        # iterate through the transformer blocks
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out