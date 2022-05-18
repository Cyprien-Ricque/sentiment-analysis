from time import sleep
import copy
import numpy as np

import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn

from tqdm import tqdm

from GRU import GRU
from MHA import TransformerModel
from RNN import RNN
from LSTM import LSTM
from test import Evaluator
from textdataset import TextDataset


class Trainer:
    def __init__(self, model: nn.Module, data: TextDataset, device: str, evaluator: Evaluator, batch_size: int = 100, lr: float = 0.001, weight_decay: float = 0, clip: float = 5.0, verbose=True):
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = clip

        self.data: TextDataset = data
        self.evaluator: Evaluator = evaluator
        self.device = device

        self.num_class = self.data.output_size
        if verbose:
            print("Num class:", self.num_class)

        self.model = model
        self.criterion = nn.CrossEntropyLoss().to(device)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def fit(self, epochs=10, starting_epoch=0):
        self.model.train()

        update_acc_frq = 100

        val_set = self.data.val_set
        train_set_sample = self.data.train_set_sample
        best_val_acc = 0
        best_model = None

        for e in range(starting_epoch, starting_epoch + epochs):
            train_losses = []
            batches = self.data.batches(self.batch_size, shuffle=False)

            train_acc = self.evaluator.accuracy(*train_set_sample, set_name='train', batch_size=500)  # Keep track and print
            val_acc = self.evaluator.accuracy(*val_set, set_name='val', batch_size=500)  # Keep track and print

            train_loss = self.evaluator.loss(*train_set_sample, criterion=self.criterion, set_name='train', batch_size=500)  # Keep track
            val_loss = self.evaluator.loss(*val_set, criterion=self.criterion, set_name='val', batch_size=500)  # Keep track

            if val_acc > best_val_acc:
                best_model = copy.deepcopy(self.model)
                best_val_acc = val_acc

            if e == starting_epoch:
                print('first val accuracy', val_acc, flush=True)
                print('first train accuracy', train_acc, flush=True)

            batches_iter = tqdm(enumerate(batches), position=0, leave=True, total=len(batches))
            batches_iter.set_description(
                f"Ep.{e} Train R_Loss: {np.mean(train_losses[-len(batches):]) if len(train_losses) > 0 else -1:.3f} Train Loss: {train_loss:.2f} Val Loss: {val_loss:.2f} Train Acc: {train_acc:.2f} Val Acc: {val_acc:.2f}")

            for i, (inputs, labels) in batches_iter:

                self.optimizer.zero_grad()

                output, hidden = self.model(inputs)

                loss = self.criterion(output, labels)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()

                batch_acc = torch.sum(torch.argmax(output, dim=1) == labels) / inputs.shape[1]

                if i % update_acc_frq == 0 and i != 0:
                    train_loss = self.evaluator.loss(*train_set_sample, criterion=self.criterion, batch_size=500)  # Print
                    val_loss = self.evaluator.loss(*val_set, criterion=self.criterion, batch_size=500)  # Print

                    train_acc = self.evaluator.accuracy(*train_set_sample, batch_size=500)  # To print
                    val_acc = self.evaluator.accuracy(*val_set, batch_size=500)  # To print

                    train_set_sample = self.data.train_set_sample  # Get new sample

                train_losses.append(loss.item())
                batches_iter.set_description(
                    f"Ep.{e} Batch acc {batch_acc:.3f} Train R_Loss: {np.mean(train_losses[-len(batches):]) if len(train_losses) > 0 else -1:.3f} Train Loss: {train_loss:.2f} Val Loss: {val_loss:.2f} Train Acc: {train_acc:.2f} Val Acc: {val_acc:.2f}")

        print('\nSet model to be the best model\n', flush=True)
        self.model = best_model
        self.evaluator.model = self.model


if __name__ == '__main__':
    use_cuda = True
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print("Use device", device)

    data: TextDataset = TextDataset(include_subtrees=False, mode='one_hot', device=device)
    print('TEXT DATASET CREATED')

    # Tunable
    hidden_size = 256
    n_layers = 3
    n_head = 12
    dropout = 0.2
    lr = 0.0001
    weight_decay = 0.00005
    batch_size = 100
    bidirectional = True
    clip = 5

    model_type = GRU

    model = model_type(device=device, input_size=data.embeddings.shape[1], output_size=data.output_size, w2v_vectors=data.embeddings,
                       hidden_size=hidden_size, n_layers=n_layers, dropout=dropout,
                       bidirectional=bidirectional
                       ).to(device)
    print('Transformer CREATED')

    evaluator = Evaluator(model=model, data=data)
    print('Evaluator CREATED')

    trainer = Trainer(
        model, data, device, evaluator=evaluator,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        clip=clip,
        verbose=True
    )
    print('Trainer CREATED')

    trainer.fit(60)
    evaluator.final_eval()
