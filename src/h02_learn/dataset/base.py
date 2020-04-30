from abc import ABC, abstractmethod
from collections import Counter

import torch
from torch.utils.data import Dataset
import numpy as np

class BaseDataset(Dataset, ABC):
    # pylint: disable=no-member

    def __init__(self, data, folds, max_tokens=None):
        self.data = data
        self.folds = folds
        self.max_tokens = max_tokens
        self.process_train(data)
        self.process_eval(data)
        self._train = True

    @staticmethod
    def subsample(word_freqs, max_tokens):
        words, weights = zip(*word_freqs)
        probs = np.array(weights)/sum(weights)
        sample = np.random.choice(words, p=probs, size=max_tokens)
        sample_counts = Counter(sample)
        sample_freqs = list(sample_counts.items())
        return sample_freqs

    @abstractmethod
    def process_train(self, data):
        pass

    @abstractmethod
    def process_eval(self, data):
        pass

    def get_word_idx(self, word):
        return [self.alphabet.char2idx('SOS')] + \
                self.alphabet.word2idx(word) + \
                [self.alphabet.char2idx('EOS')]

    def __len__(self):
        if self._train:
            return self.train_instances
        return self.eval_instances

    def __getitem__(self, index):
        word_indices = self.word_train[index] if self._train else self.word_eval[index]
        weight = torch.Tensor([1]) if self._train else self.weights[index]
        token = self.token_train[index] if self._train else self.token_eval[index]
        # return token here
        return (word_indices, weight, index, token)

    def train(self):
        self._train = True

    def eval(self):
        self._train = False
