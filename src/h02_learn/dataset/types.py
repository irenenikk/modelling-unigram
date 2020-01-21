import torch

from .base import BaseDataset


class TypeDataset(BaseDataset):

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
