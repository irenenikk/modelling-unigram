from rangetree import RangeTree

import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, data, folds):
        self.data = data
        self.folds = folds
        self.process_data(data)

    def process_data(self, data):
        folds_data, alphabet, _ = data
        self.alphabet = alphabet

        self.word_freqs = [(word, info['count'])
                           for fold in self.folds
                           for word, info in folds_data[fold].items()]
        self.word_idxs, self.n_instances = self.build_token_list(self.word_freqs)

    def build_token_list(self, word_freqs):
        begin, end = 0, 0
        word_idxs = RangeTree()
        for word, freq in word_freqs:
            end += freq
            word_idxs[begin:end] = torch.LongTensor(self.get_word_idx(word))
            begin = end

        return word_idxs, end

    def get_word_idx(self, word):
        return [self.alphabet.char2idx('SOS')] + \
            self.alphabet.word2idx(word) + \
            [self.alphabet.char2idx('EOS')]

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return self.word_idxs[index]
