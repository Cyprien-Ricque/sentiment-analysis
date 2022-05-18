import torch
import torch.nn as nn
from torch.nn import functional as f

import numpy as np
import pandas as pd
import math
from tqdm import tqdm

from gensim.models import KeyedVectors

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import treebank
from nltk.corpus.reader import StreamBackedCorpusView
from nltk.tree.tree import Tree

from model import RNN
from test import Evaluator

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


nltk.download('treebank')


class DataLoader:
    folder = './data/trainDevTestTrees_PTB/'
    train_file = folder + 'train.txt'
    test_file = folder + 'test.txt'

    def __init__(self, device, labels_linear=False, include_subtrees=False, verbose=False, equalize_train_set_cats=False):
        self.X_train_not_eq = None
        self.y_train_not_eq = None
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = None, None, None, None, None, None
        self.X_train_before_pad, self.X_test_before_pad, self.X_val_before_pad = None, None, None
        self.include_subtrees = include_subtrees
        self.labels_linear = labels_linear
        self.equalize_train_set_cats = equalize_train_set_cats

        if verbose:
            print('download datasets')

        # Parse data files
        self.test_set: StreamBackedCorpusView = treebank.parsed_sents(self.test_file)
        self.train_set: StreamBackedCorpusView = treebank.parsed_sents(self.train_file)
        self.test_set._encoding = 'utf-8'
        self.train_set._encoding = 'utf-8'
        self.verbose = verbose
        self.device = device

        if verbose:
            print("First tree leaves test set")
            print(self.test_set[0].leaves())
            print("First tree leaves train set")
            print(self.train_set[0].leaves())

    def drop_empty(self, embedded, labels):
        """
        Drop values when the embedded value is empty
        Warning: make sure nothing is dropped in test set
        """
        non_zero_idx = [i for i, ebd in enumerate(embedded) if len(ebd) != 0]
        if self.verbose:
            print("drop zeros:", len(embedded) - len(non_zero_idx))

        embedded = [embedded[i] for i in non_zero_idx]
        labels = np.array([labels[i] for i in non_zero_idx])
        return embedded, labels

    def createSets(self, key_to_index, train_val_split=0.95, pad=25):
        """
        Creates all needed datasets with the correct format from StreamBackedCorpusView.
        """
        self.X_train_before_pad, self.y_train_not_eq = self.encode(self.train_set[:int(len(self.train_set) * train_val_split)], key_to_index, include_subtrees=self.include_subtrees)
        self.X_val_before_pad, self.y_val = self.encode(self.train_set[int(len(self.train_set) * train_val_split):], key_to_index)
        self.X_test_before_pad, self.y_test = self.encode(self.test_set, key_to_index)

        if self.verbose and self.include_subtrees is False:
            train_val = self.X_train_before_pad + self.X_val_before_pad
            print("mean missing words per sentence", np.mean(np.array(list(map(lambda x: len(x.leaves()), self.train_set))) - np.array(list(map(lambda x: len(x), train_val)))))
            print("mean words per sentence original", np.mean(np.array(list(map(lambda x: len(x.leaves()), self.train_set)))))
            print("mean words per sentence embedded", np.mean(np.array(list(map(lambda x: len(x), train_val)))))

        self.X_train_before_pad, self.y_train_not_eq = self.drop_empty(self.X_train_before_pad, self.y_train_not_eq)
        self.X_val_before_pad, self.y_val = self.drop_empty(self.X_val_before_pad, self.y_val)
        self.X_test_before_pad, self.y_test = self.drop_empty(self.X_test_before_pad, self.y_test)

        self.y_train_not_eq = torch.LongTensor(self.y_train_not_eq.astype(int)).to(self.device)
        self.y_test = torch.LongTensor(self.y_test.astype(int)).to(self.device)
        self.y_val = torch.LongTensor(self.y_val.astype(int)).to(self.device)

        if self.labels_linear is False:
            self.y_train_not_eq = f.one_hot(self.y_train_not_eq).to(float)
            self.y_test = f.one_hot(self.y_test).to(float)
            self.y_val = f.one_hot(self.y_val).to(float)
        else:
            self.y_train_not_eq = self.y_train_not_eq.to(torch.float32)
            self.y_test = self.y_test.to(torch.float32)
            self.y_val = self.y_val.to(torch.float32)

        self.pad(pad)

    def pad(self, pad):
        self.X_train_not_eq = padFeatures(self.X_train_before_pad, pad)
        self.X_val = padFeatures(self.X_val_before_pad, pad)
        self.X_test = padFeatures(self.X_test_before_pad, pad)

        self.X_train_not_eq = torch.tensor(self.X_train_not_eq).to(self.device)
        self.X_val = torch.tensor(self.X_val).to(self.device)
        self.X_test = torch.tensor(self.X_test).to(self.device)

        if self.verbose:
            print('X_train', self.X_train_not_eq.shape, 'y_train', self.y_train_not_eq.shape)
            print('X_val', self.X_val.shape, 'y_val', self.y_val.shape)
            print('X_test', self.X_test.shape, 'y_test', self.y_test.shape)

        self.X_train, self.y_train = self.X_train_not_eq, self.y_train_not_eq

        if self.equalize_train_set_cats is True:
            self.X_train, self.y_train = self.equalize_set(self.X_train_not_eq, self.y_train_not_eq)

    def encode(self, df, key_to_index, include_subtrees=False):
        """
        Encode words using the provided dictionary (key_to_index)
        """
        embedded = []
        labels = []

        if not include_subtrees:
            for tree in tqdm(df, total=len(df)):
                embedded.append([key_to_index[w] for w in tree.leaves() if w in key_to_index])
                labels.append(tree.label())
        else:
            for tree in tqdm(df, total=len(df)):
                for subtree in tree.subtrees():
                    embedded.append([key_to_index[w] for w in subtree.leaves() if w in key_to_index])
                    labels.append(subtree.label())
        if self.verbose:
            print('embedded[0]:', embedded[0])

        return embedded, labels

    def equalize_set(self, X, y):
        """
        This function duplicates X y value to make the set have the same number of value in each label classes
        """
        logger.debug(f'equalize_set {X.shape} {y.shape}')
        if self.labels_linear:
            logger.error('Equalize set does not support self.labels_linear = True')
            return X, y
        most_common_cat = int(torch.mode(torch.argmax(y, axis=1), axis=0).values.cpu())
        logger.info(f'Most common category is {most_common_cat}')
        target = int(torch.sum(torch.argmax(y, axis=1) == most_common_cat).cpu())

        new_X_train = torch.tensor([]).cuda()
        new_y_train = torch.tensor([]).cuda()

        for c in range(y.shape[1]):
            _c_idx = torch.argmax(y, axis=1) == c
            logger.debug(f'_c_idx {len(_c_idx)}')
            x_c = X[_c_idx, :]
            y_c = y[_c_idx, :]
            print(x_c.shape, y_c.shape)

            idx_selection = torch.randperm(x_c.shape[0])[:target - x_c.shape[0] * int(target / x_c.shape[0])]

            new_X_train = torch.concat((new_X_train,
                                        x_c.repeat(int(target / x_c.shape[0]), 1),
                                        x_c[idx_selection]
                                        ))
            new_y_train = torch.concat((new_y_train,
                                        y_c.repeat(int(target / y_c.shape[0]), 1),
                                        y_c[idx_selection]
                                        ))
            print(new_X_train.shape, new_y_train.shape)

        shuffle = torch.randperm(new_X_train.shape[0])
        new_X_train = new_X_train[shuffle].to(torch.int32)
        new_y_train = new_y_train[shuffle]
        return new_X_train, new_y_train


class Vocabulary:
    """
    Basic vocabulary class
    can compute vocabulary from DataLoader class
    """
    def __init__(self, data: DataLoader):
        self._index_to_key = []

        for row in data.train_set:
            self._index_to_key += row.leaves()
            self._index_to_key = list(set(self._index_to_key))

        self._key_to_index = {a: b for b, a in enumerate(self._index_to_key)}

    def __len__(self):
        return len(self._key_to_index)

    @property
    def key_to_index(self):
        return self._key_to_index

    @property
    def index_to_key(self):
        return self._key_to_index


class WordVectors:
    """
    load pre-trained word2vec and provide easy to use function to work with it
    """

    file = './data/glove.6B.300d.txt'

    def __init__(self, limit=None, verbose=True):
        if verbose:
            print('download word2vec', self.file)
        self.wv = KeyedVectors.load_word2vec_format(self.file, binary=False, no_header=True, limit=limit)
        self.dict = self.wv.index_to_key
        self.shape = self.vectors().shape

        assert self.shape[0] == len(self), f'Invalid dict size, {self.shape[0]} must match {len(self)}'
        if verbose:
            print(f'Word2Vec shape {self.shape}')

    def vectors(self):
        return self.wv.vectors

    def __len__(self):
        return len(self.wv.key_to_index)

    def key_to_index(self):
        return self.wv.key_to_index

    def index_to_key(self):
        return self.dict


def padFeatures(embeds, seq_length):
    """
    Make all embeds the same length by cutting large vectors and filling shorts ones with 0s
    """
    features = np.zeros((len(embeds), seq_length), dtype=int)

    for i, row in enumerate(embeds):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


import random


class Batches:
    """
    Creates batches on X and y according to give parameters.
    allows to iter on it
    """
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.nb_batches = math.ceil(self.X.shape[0] / batch_size)
        self.shuffle = shuffle

    def __iter__(self):
        batches_order = list(range(self.nb_batches))
        if self.shuffle:
            random.shuffle(batches_order)
        for batch in batches_order:
            yield self.X[batch * self.batch_size:(batch + 1) * self.batch_size], self.y[batch * self.batch_size:(batch + 1) * self.batch_size]

    def __len__(self):
        return self.nb_batches


class Trainer:
    """
    Combines all moodule to train the model with specific training parameters.
    """
    def __init__(self, model, data, device, evaluator, batch_size=100, lr=0.0001, weight_decay=.0, clip=5, labels_linear=False, verbose=True):
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = clip
        self.__labels_linear = labels_linear

        self.data: DataLoader = data
        self.evaluator: Evaluator = evaluator

        self.num_class = self.data.y_test.shape[1] if len(self.data.y_test.shape) > 1 else 1
        if verbose:
            print("Num class:", self.num_class, torch.unique(self.data.y_test))

        self.model = model
        if not self.__labels_linear:
            self.criterion = nn.CrossEntropyLoss().to(device)
        else:
            self.criterion = nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def fit(self, epochs=10, starting_epoch=0):
        """
        Train the model

        """
        self.model.train()

        update_acc_frq = 10
        X_train, y_train, X_val, y_val = self.data.X_train, self.data.y_train, self.data.X_val, self.data.y_val

        for e in range(starting_epoch, starting_epoch + epochs):
            train_losses = []
            batches = Batches(X_train, y_train, self.batch_size)  # Create batches
            batches_iter = tqdm(enumerate(batches), position=0, leave=True, total=len(batches))  # Create iterator over batches with progress bar

            train_acc = self.evaluator.accuracy(X_train, y_train, set_name='train')  # Keep track and print
            val_acc = self.evaluator.accuracy(X_val, y_val, set_name='val')  # Keep track and print
            self.evaluator.loss(X_train, y_train, criterion=self.criterion, set_name='train')  # Keep track
            self.evaluator.loss(X_val, y_val, criterion=self.criterion, set_name='val')  # Keep track

            batches_iter.set_description(f"Ep.{e} Train R_Loss: {np.mean(train_losses) if len(train_losses) else 99:.3f} Train Acc: {train_acc:.2f} Val Acc: {val_acc:.2f}")

            for i, (inputs, labels) in batches_iter:
                self.model.zero_grad()

                output, hidden = self.model(inputs)

                loss = self.criterion(output.squeeze(), labels)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()

                if i % update_acc_frq == 0 and i != 0:
                    train_acc = self.evaluator.accuracy(X_train, y_train)  # To print
                    val_acc = self.evaluator.accuracy(X_val, y_val)  # To print

                train_losses.append(loss.item())
                batches_iter.set_description(f"Ep.{e} Train R_Loss: {np.mean(train_losses[-10:]):.3f} Train Acc: {train_acc:.2f} Val Acc: {val_acc:.2f}")


if __name__ == '__main__':
    use_cuda = True
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

    wv: WordVectors = WordVectors(limit=None, verbose=True)  # Load pre-trained w2v
    data: DataLoader = DataLoader(device, labels_linear=False, include_subtrees=False, verbose=True, equalize_train_set_cats=False)  # Load train / test sets

    # Some Tunable parameters
    hidden_size = 256
    n_layers = 1
    pad = 50
    epochs = 200

    data.createSets(key_to_index=wv.key_to_index(), train_val_split=0.93, pad=pad)  # Create datasets usable by the model

    model = RNN(device=device,
                input_size=wv.shape[1],
                output_size=data.y_test.shape[1],
                hidden_size=hidden_size,
                n_layers=n_layers,
                w2v_vectors=torch.tensor(wv.vectors()), dropout=0.5, freeze=False).to(device)  # Create the model
    evaluator = Evaluator(model=model, train=(data.X_train, data.y_train), test=(data.X_test, data.y_test), val=(data.X_val, data.y_val))

    ## Create the trainer model that combines everything we created before to train and evaluate the model
    trainer = Trainer(
        model, data, device, evaluator=evaluator,
        batch_size=100,
        lr=0.0001,
        weight_decay=0.003,
        clip=5,
        verbose=True
    )

    trainer.fit(epochs)  # Start training on "epochs" number of epochs
    evaluator.final_eval()  # Perform a final evaluation on all sets and plot the accuracy / lost of train and val set.

    """
    To work with improved version set DataLoader hyper parameters
    include_subtrees and equalize_train_set_cats to True
    
    """