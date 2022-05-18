import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, device, output_size: int, input_size: int, w2v_vectors, dim_feedforward: int = 3072, n_layers: int = 12, n_head: int = 12, dropout: float = 0.1, freeze: bool = True):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_size, 0)
        encoder_layers = TransformerEncoderLayer(input_size, n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.embedding = nn.Embedding.from_pretrained(w2v_vectors, freeze=freeze)
        self.input_size = input_size
        self.fc = nn.Linear(input_size, output_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tuple[Tensor, None]:
        src = self.embedding(src) * math.sqrt(self.input_size)
        src = self.pos_encoder(src)
        output_enc = self.transformer_encoder(src)

        avg = output_enc.mean(dim=0)

        output = self.fc(avg)

        return output, None

    def predict(self, X, to_class=False):
        output, _ = self(X)
        if to_class:
            return torch.argmax(output.cpu().detach(), dim=1)
        return output.cpu().detach()
