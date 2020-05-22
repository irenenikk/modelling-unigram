from collections import Counter
import copy

import torch
from torch.utils.data import Dataset
import numpy as np

from h02_learn.dataset.base import get_folds_data, get_word_idx
#from h02_learn.dataset.tokens import build_token_list


class SentenceDataset(Dataset):
    # pylint: disable=no-member

    def __init__(self, data, folds, max_tokens=None):
        self.data = data
        self.folds = folds
        self.max_tokens = max_tokens
        self.sentences = self.get_folds_data(data)

    def get_folds_data(self, data):
        folds_data, sentence_data, alphabet, _ = data
        self.alphabet = copy.deepcopy(alphabet)
        # add whitespace to vocabulary
        self.alphabet.add_word(' ')
        all_sentences = [sent
                      for fold in self.folds
                      for sent in sentence_data[fold].items()]

        sentences = []
        if self.max_tokens is None:
            total_tokens = len([word for sentence in all_sentences for word in sentence])
            self.max_tokens = total_tokens
        n_tokens = 0
        i = 0
        while (n_tokens < self.max_tokens):
            sentence = all_sentences[i]
            sentences.append(sentence)
            n_tokens += len(sentence)
            i += 1
        print('Using', len(sentences), 'sentences')
        return sentences

    def __len__(self):
        return len(sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        indices = torch.LongTesor([self.alphabet.word2idx('SOS')])
        indices += torch.LongTesor([self.alphabet.word2idx(word) + self.alphabet.word2idx(' ')
                   for word in sentence])
        indices[-1] = torch.LongTesor(self.alphabet.word2idx('EOS'))
        return (indices, torch.Tensor(1), index, '')
