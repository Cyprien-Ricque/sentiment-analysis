import torch
import matplotlib.pyplot as plt
from textdataset import TextDataset
import logging
import torch.nn.functional as F


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Evaluator:
    def __init__(self, model, data: TextDataset):
        self.model = model
        self.data: TextDataset = data

        self.accuracies = {}
        self.losses = {}

    def accuracy(self, X, y, set_name=None, set_train_mode=True, batch_size=None):
        self.model.eval()
        y_pred = None
        if batch_size is not None:
            for i in range(0, X.size(1), batch_size):
                y_pred = self.model.predict(X[:, i:i+batch_size], to_class=True).cpu() if y_pred is None else torch.concat((y_pred, self.model.predict(X[:, i:i+batch_size], to_class=True).cpu()))
        else:
            y_pred = self.model.predict(X, to_class=True)

        acc = torch.sum(y_pred == y.cpu()) / y_pred.shape[0]

        # Save accuracy in history
        if set_name is not None:
            if set_name not in self.accuracies:
                self.accuracies[set_name] = []
            self.accuracies[set_name].append(acc)
        if set_train_mode:
            self.model.train()
        return acc

    def loss(self, X, y, criterion=None, set_name=None, set_train_mode=True, batch_size=None):
        self.model.eval()

        y_hat = None
        if batch_size is not None:
            for i in range(0, X.size(1), batch_size):
                y_hat = self.model.predict(X[:, i:i+batch_size], to_class=False).cpu() if y_hat is None else torch.concat((y_hat, self.model.predict(X[:, i:i+batch_size], to_class=False).cpu()))
        else:
            y_hat = self.model.predict(X.detach(), to_class=False).cpu()

        loss = F.cross_entropy(y_hat.detach(), y.detach().cpu())
        if criterion:
            criterion.zero_grad()
        loss = loss.detach()

        # Save loss in history
        if set_name is not None:
            if set_name not in self.losses:
                self.losses[set_name] = []
            self.losses[set_name].append(loss)
        if set_train_mode:
            self.model.train()
        return loss

    def final_eval(self, compute_final_scores=True):
        if compute_final_scores:
            self.model.eval()
            self.accuracy(*self.data.test_set, set_name='test', batch_size=500)
            self.accuracy(*self.data.train_set_sample, set_name='train', batch_size=500)
            self.accuracy(*self.data.val_set, set_name='val', batch_size=500)
            self.loss(*self.data.train_set_sample, set_name='train', batch_size=500)
            self.loss(*self.data.val_set, set_name='val', batch_size=500)

        print('Test Accuracy', self.accuracies['test'][-1])
        print('Train Accuracy', self.accuracies['train'][-1])
        print('Val Accuracy', self.accuracies['val'][-1])

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 12))

        title_size = 15
        label_size = 13

        ax1.plot(self.accuracies['train'], label='training')
        ax1.plot(self.accuracies['val'], label='validation')
        ax1.legend()
        ax1.set_title('Accuracy over epochs', fontsize=title_size)
        ax1.set_xlabel('Epochs', fontsize=label_size)
        ax1.set_ylabel('Accuracy', fontsize=label_size)

        ax2.plot(self.losses['train'], label='training')
        ax2.plot(self.losses['val'], label='validation')
        ax2.legend()
        ax2.set_title('Loss over epochs', fontsize=title_size)
        ax2.set_xlabel('Epochs', fontsize=label_size)
        ax2.set_ylabel('Loss', fontsize=label_size)

        plt.show()

        return float(self.accuracies['test'][-1]), float(self.accuracies['train'][-1]), float(self.accuracies['val'][-1])
