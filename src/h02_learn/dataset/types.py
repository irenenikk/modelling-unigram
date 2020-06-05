import torch

from .base import BaseDataset


class TypeDataset(BaseDataset):

    def process_train(self, data):
        word_freqs = self.get_folds_data(data)
        self.token_train = [word for word, _ in word_freqs]
        self.word_train = [torch.LongTensor(self.get_word_idx(word)) for word in self.token_train]
        self.train_instances = len(self.word_train)

    def process_eval(self, data):
        self.word_eval = self.word_train
        self.token_eval = self.token_train
        self.weights = [torch.Tensor([1])] * len(self.word_eval)
        self.eval_instances = len(self.word_eval)
