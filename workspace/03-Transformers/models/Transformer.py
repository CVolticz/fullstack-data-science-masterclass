import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from TransformerBlock import TransformerBlock

class Transformer(nn.Module):
    """
        Class Module to Construct the Transformer Block
    """
    def __init__(self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100
    ):
        super(Transformer, self).__init__()
        
        # instantiate the encoder and decoder
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device


    def make_src_mask(self, src):
        """
            Method to make the source mask
        """
        # src shape: (N, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len)

        return src_mask.to(self.device)
    

    def make_trg_mask(self, trg):
        """
            Method to make the target mask
        """
        # trg shape: (N, trg_len)
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        # trg_mask shape: (N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)
    

    def forward(self, src, trg):
        """
            Method to perform one forward pass through a batch of the transformer
            Inputs:
                src {tensor}: source tensor
                trg {tensor}: target tensor
            Outputs:
                out {tensor}: output of the transformer
        """

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # start the encoder
        enc_src = self.encoder(src, src_mask)

        # start the decoder
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the input and target tensors
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    # define the starting parameters 
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    # instantiate the transformer
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)