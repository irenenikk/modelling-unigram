import torch

from .base import BaseDataset


class TypeDataset(BaseDataset):

    def process_train(self, data):
        # folds_data, alphabet, _ = data
        # self.alphabet = alphabet

        # word_freqs = [(word, info['count'])
        #               for fold in self.folds
        #               for word, info in folds_data[fold].items()]
        # token_amount = sum([freq for _, freq in word_freqs])
        # if self.max_tokens is not None and token_amount >= self.max_tokens:
        #     word_freqs = self.subsample(word_freqs, self.max_tokens)

        word_freqs = self.get_folds_data(data)
        self.words = [word for word, _ in word_freqs]
        self.word_train = [torch.LongTensor(self.get_word_idx(word)) for word in self.words]
        self.train_instances = len(self.word_train)

    def process_eval(self, data):
        self.word_eval = self.word_train
        self.weights = [torch.Tensor([1])] * len(self.word_eval)
        self.eval_instances = len(self.word_eval)
