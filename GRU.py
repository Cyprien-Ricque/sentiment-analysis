import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class GRU(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_size, n_layers, w2v_vectors, dropout, bidirectional=False):
        super(GRU, self).__init__()
        self.device = device

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.input_size = input_size

        self.embedding = nn.Embedding.from_pretrained(w2v_vectors, freeze=True)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=hidden_size * (self.bidirectional + 1), out_features=output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.init_hidden(x.shape[1])
        embeds = self.embedding(x) * math.sqrt(self.input_size)
        embeds = self.dropout(embeds)

        rnn_out, hidden = self.gru(embeds, hidden)

        out = self.fc(rnn_out[-1, :, :])

        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers * (1 + self.bidirectional), batch_size, self.hidden_size).to(self.device)

    def predict(self, X, to_class=False):
        output, hidden = self(X)
        if to_class:
            return torch.argmax(output.cpu(), dim=1)
        return output
