import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_size, n_layers, w2v_vectors, dropout, bidirectional=False):
        super(RNN, self).__init__()
        self.device = device

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(w2v_vectors, freeze=True)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0, bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=hidden_size * (self.bidirectional + 1), out_features=output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.init_hidden(x.shape[1])
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)

        rnn_out, hidden = self.rnn.forward(embeds, hidden)

        out = self.fc(rnn_out[-1, :, :])

        return out, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers * (1 + self.bidirectional), batch_size, self.hidden_size).to(self.device))

    def predict(self, X, to_class=False):
        out, hidden = self(X)
        if to_class:
            return torch.argmax(out.cpu(), dim=1)
        return out
