import copy

import torch
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
    # pylint: disable=no-member

    def __init__(self, data, folds, max_tokens=None):
        self.data = data
        self.folds = folds
        self.max_tokens = max_tokens
        self.sentences = self.get_folds_data(data)

    def get_folds_data(self, data):
        _, sentence_data, alphabet, _ = data
        self.alphabet = copy.deepcopy(alphabet)
        # add whitespace to vocabulary
        self.alphabet.add_word(' ')
        all_sentences = [sent
                         for fold in self.folds
                         for sent in sentence_data[fold]]
        sentences = []
        if self.max_tokens is None:
            total_tokens = len([word for sentence in all_sentences for word in sentence])
            self.max_tokens = total_tokens
        n_tokens = 0
        i = 0
        while n_tokens < self.max_tokens:
            sentence = all_sentences[i]
            if sentence == []:
                i += 1
                continue
            sentences.append(sentence)
            n_tokens += len(sentence)
            i += 1
        print('Using', len(sentences), 'sentences')
        return sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        indices = [self.alphabet.char2idx('SOS')]
        indice_list = [self.alphabet.word2idx(word.lower()) + self.alphabet.word2idx(' ')
                       for word in sentence]
        indices += [index for word in indice_list for index in word]
        indices[-1] = self.alphabet.char2idx('EOS')
        return (torch.LongTensor(indices), torch.Tensor([1]), index, '')

    def train(self):
        pass

    def eval(self):
        pass
