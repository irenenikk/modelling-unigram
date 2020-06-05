from rangetree import RangeTree
import torch

from .base import BaseDataset


class TokenDataset(BaseDataset):

    def process_train(self, data):
        self.word_freqs = self.get_folds_data(data)
        self.word_train, self.token_train, self.train_instances = \
                                                self.build_token_list(self.word_freqs)

    def process_eval(self, data):
        self.word_eval = [torch.LongTensor(self.get_word_idx(word)) for word, _ in self.word_freqs]
        self.token_eval = [word for word, _ in self.word_freqs]
        self.weights = [torch.Tensor([freq]) for _, freq in self.word_freqs]
        self.eval_instances = len(self.word_eval)

    def build_token_list(self, word_freqs):
        begin, end = 0, 0
        word_idxs = RangeTree()
        tokens = RangeTree()
        for word, freq in word_freqs:
            end += freq
            word_idxs[begin:end] = torch.LongTensor(self.get_word_idx(word))
            tokens[begin:end] = word
            begin = end
        return word_idxs, tokens, end
