from rangetree import RangeTree
import torch

from .base import BaseDataset


class TokenDataset(BaseDataset):
    # def __init__(self, data, folds):
    #     self.data = data
    #     self.folds = folds
    #     self.process_data(data)

    def process_train(self, data):
        folds_data, alphabet, _ = data
        self.alphabet = alphabet

        self.word_freqs = [(word, info['count'])
                           for fold in self.folds
                           for word, info in folds_data[fold].items()]
        self.word_train, self.train_instances = self.build_token_list(self.word_freqs)

    def process_eval(self, data):
        self.word_eval = [torch.LongTensor(self.get_word_idx(word)) for word, _ in self.word_freqs]
        self.weights = [torch.Tensor([freq]) for _, freq in self.word_freqs]
        self.eval_instances = len(self.word_eval)

    def build_token_list(self, word_freqs):
        begin, end = 0, 0
        word_idxs = RangeTree()
        for word, freq in word_freqs:
            end += freq
            word_idxs[begin:end] = torch.LongTensor(self.get_word_idx(word))
            begin = end

        return word_idxs, end

    # def get_word_idx(self, word):
    #     return [self.alphabet.char2idx('SOS')] + \
    #         self.alphabet.word2idx(word) + \
    #         [self.alphabet.char2idx('EOS')]

    # def __len__(self):
    #     return self.n_instances

    # def __getitem__(self, index):
    #     return self.word_idxs[index]
