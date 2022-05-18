from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import torch
import math
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Batches:
    def __init__(self, it, length):
        self.it = it
        self.length = length

    def __iter__(self):
        for x, y in self.it:
            yield x, y

    def __len__(self):
        return self.length


class TextDataset:
    def __init__(self, include_subtrees=False, mode='one_hot', device='cpu'):
        """

        :param mode: in ('one_hot', 'linear')
        :param device:
        """
        if include_subtrees is True:
            logger.warning('You can\'t use subtrees for this HW')

        self.mode = mode
        self.device = device

        # set up fields
        self.TEXT = data.Field()
        self.LABEL = data.Field(sequential=False, dtype=torch.long)

        self.train, self.val, self.test = datasets.SST.splits(self.TEXT, self.LABEL, fine_grained=True, train_subtrees=include_subtrees)

        # build the vocabulary
        # you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
        # self.TEXT.build_vocab(self.train, vectors=Vectors(name='vector.txt', cache='./homework5/data'))
        self.TEXT.build_vocab(self.train, vectors='glove.6B.300d')
        self.LABEL.build_vocab(self.train)

        print(self.TEXT.vocab.itos[:10])
        print(self.LABEL.vocab.stoi)
        print(self.TEXT.vocab.freqs.most_common(20))

        # print vocab information
        print('len(self.TEXT.vocab)', len(self.TEXT.vocab))
        print('self.TEXT.vocab.vectors.size()', self.TEXT.vocab.vectors.size())

        # make iterator for splits
        self._train_set, self._val_set, self._test_set = data.BucketIterator.splits((self.train, self.val, self.test),
                                                                                    batch_sizes=(len(self.train), len(self.val), len(self.test)))

        self._test_set_buf = None
        self._val_set_buf = None
        self._train_set_buf = None

        iter_ = next(iter(self._train_set))
        self.__X_train = iter_.text
        self.__y_train = iter_.label


    def batches(self, batch_size, **kwargs):
        """
        Iterate over the dataset for training
        :return: X_train, y_train
        """

        length = math.ceil(len(self.train) / batch_size)
        return Batches(it=((self.train_set[0][:, i:i + batch_size], self.train_set[1][i:i + batch_size]) for i in range(0, self.train_set[0].shape[1], batch_size)),
                       length=length)


    def __X_format(self, X):
        return X.to(self.device)

    def __y_mode_format(self, y):
        if self.mode == 'one_hot':
            return self.__y_to_one_hot(y)
        if self.mode == 'linear':
            return self.__y_to_linear(y)
        return y.to(self.device).to(torch.float)

    def __y_to_one_hot(self, y):
        return (y - 1).to(self.device)

    def __y_to_linear(self, y):
        mapping = {
            'neutral': 0,
            'positive': .5,
            'negative': -.5,
            'very positive': 1,
            'very negative': -1
        }

        def to_linear(x):
            return mapping[self.LABEL.vocab.itos[x]]

        f = np.vectorize(to_linear)
        return f(y)

    @property
    def val_set(self):
        if self._val_set_buf:
            return self._val_set_buf
        full_iter_ = next(iter(self._val_set))
        self._val_set_buf = (self.__X_format(full_iter_.text),
                             self.__y_mode_format(full_iter_.label))
        return self._val_set_buf

    @property
    def test_set(self):
        if self._test_set_buf:
            return self._test_set_buf
        full_iter_ = next(iter(self._test_set))
        self._test_set_buf = (self.__X_format(full_iter_.text),
                              self.__y_mode_format(full_iter_.label))
        return self._test_set_buf

    @property
    def train_set(self):
        if self._train_set_buf:
            return self._train_set_buf
        full_iter_ = next(iter(self._train_set))
        self._train_set_buf = (self.__X_format(full_iter_.text),
                               self.__y_mode_format(full_iter_.label))
        return self._train_set_buf

    @property
    def output_size(self):
        return dict(
            one_hot=5,
            linear=1
        )[self.mode]

    @property
    def train_set_sample(self):
        limit = 1000
        if len(self.train) <= limit:
            return self.train_set
        idx = torch.randperm(len(self.train))
        X = self.__X_format(self.__X_train[:, idx[:limit]])
        y = self.__y_mode_format(self.__y_train[idx[:limit]])
        return X, y

    @property
    def y_val(self):
        if self.mode == 'one_hot':
            return self.__y_to_one_hot(next(iter(self.val_set)).label)
        if self.mode == 'linear':
            return self.__y_to_linear(next(iter(self.val_set)).label)
        return self.y_val_raw

    @property
    def y_test_raw(self):
        return next(iter(self.test_set)).label.to(self.device).to(torch.float)

    @property
    def y_train_raw(self):
        return next(iter(self.train_set)).label.to(self.device).to(torch.float)

    @property
    def y_val_raw(self):
        return next(iter(self.val_set)).label.to(self.device).to(torch.float)

    @property
    def embeddings(self):
        return self.TEXT.vocab.vectors.to(self.device)


if __name__ == '__main__':
    td = TextDataset()
