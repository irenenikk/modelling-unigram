import torch
# from torch.utils.data import Dataset

from .base import BaseDataset


class TypeDataset(BaseDataset):
    # def __init__(self, data, folds):
    #     self.data = data
    #     self.folds = folds
    #     self.process_data(data)
    #     self.n_instances = len(self.words)

    def process_train(self, data):
        folds_data, alphabet, _ = data
        self.alphabet = alphabet

        self.words = [word for fold in self.folds for word in folds_data[fold].keys()]
        self.word_train = [torch.LongTensor(self.get_word_idx(word)) for word in self.words]
        self.train_instances = len(self.word_train)

    def process_eval(self, data):
        self.word_eval = self.word_train
        self.weights = [1] * len(self.word_eval)
        self.eval_instances = len(self.word_eval)

    # def get_word_idx(self, word):
    #     return [self.alphabet.char2idx('SOS')] + \
    #         self.alphabet.word2idx(word) + \
    #         [self.alphabet.char2idx('EOS')]

    # def __len__(self):
    #     return self.n_instances

    # def __getitem__(self, index):
    #     return self.word_idxs[index]
