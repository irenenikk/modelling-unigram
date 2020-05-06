import torch

from torch.utils.data import Dataset

class TableLabelDataset(Dataset):
    # pylint: disable=no-member

    def __init__(self, tables_with_word_labels, alphabet):
        # maps a token to the indices having it as the label
        self.alphabet = alphabet
        self.dampened_tokens = []
        for token, tables in tables_with_word_labels.items():
            self.dampened_tokens += [token] * len(tables)
        print('Created a table label dataset of size', len(self.dampened_tokens))

    def __len__(self):
        return len(self.dampened_tokens)

    def __getitem__(self, index):
        word = self.dampened_tokens[index]
        word_indices = torch.LongTensor([self.alphabet.char2idx('SOS')] + \
                                        self.alphabet.word2idx(word) + \
                                        [self.alphabet.char2idx('EOS')])
        weight = torch.Tensor([1])
        return (word_indices, weight, index)
