import torch

from torch.utils.data import Dataset

class TableLabelDataset(Dataset):
    # pylint: disable=no-member

    def __init__(self, tables_with_word_labels, alphabet):
        self.alphabet = alphabet
        self.dampened_tokens = []
        for token, table_amount in tables_with_word_labels.items():
            self.dampened_tokens += [token] * table_amount
        print('Created a table label dataset of size', len(self.dampened_tokens))

    def __len__(self):
        return len(self.dampened_tokens)

    def __getitem__(self, index):
        word = self.dampened_tokens[index]
        word_indices = torch.LongTensor([self.alphabet.char2idx('SOS')] + \
                                        self.alphabet.word2idx(word) + \
                                        [self.alphabet.char2idx('EOS')])
        weight = torch.Tensor([1])
        return (word_indices, weight, index, word)
