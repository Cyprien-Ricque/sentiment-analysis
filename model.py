import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    """
    Pytorch model that implements word embedding and RNN layer
    """
    def __init__(self, device, input_size, output_size, hidden_size, n_layers, dropout, w2v_vectors=None, voc_size=None, freeze=True):
        super(RNN, self).__init__()
        self.device = device

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        if w2v_vectors is not None:
            self.embedding = nn.Embedding.from_pretrained(w2v_vectors, freeze=freeze)
        else:
            self.embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=input_size)  # Load word2vec from pre-trained

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout, nonlinearity='relu')
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        hidden = self.init_hidden(x.shape[0])
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)

        rnn_out, hidden = self.rnn(embeds, hidden)

        out = self.fc(rnn_out[:, -1, :])

        return out, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device))


