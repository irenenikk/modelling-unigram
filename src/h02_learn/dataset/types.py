import torch
from torch.utils.data import Dataset


class TypeDataset(Dataset):
    def __init__(self, data, folds):
        self.data = data
        self.folds = folds
        self.process_data(data)
        self.n_instances = len(self.words)

    def process_data(self, data):
        folds_data, alphabet, _ = data
        self.alphabet = alphabet

        self.words = [word for fold in self.folds for word in folds_data[fold].keys()]
        self.word_idxs = [torch.LongTensor(self.get_word_idx(word)) for word in self.words]

    def get_word_idx(self, word):
        return [self.alphabet.char2idx('SOS')] + \
            self.alphabet.word2idx(word) + \
            [self.alphabet.char2idx('EOS')]

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return self.word_idxs[index]
