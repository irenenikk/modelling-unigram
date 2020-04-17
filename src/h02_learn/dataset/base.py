from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    # pylint: disable=no-member

    def __init__(self, data, folds, sample_size=None):
        self.data = data
        self.folds = folds
        self.process_train(data)
        self.process_eval(data)
        self._train = True
        self.sample_size = sample_size

    @abstractmethod
    def process_train(self, data):
        pass

    @abstractmethod
    def process_eval(self, data):
        pass

    def get_word_idx(self, word):
        return [self.alphabet.char2idx('SOS')] + \
                self.alphabet.word2idx(word) + \
                [self.alphabet.char2idx('EOS')]

    def __len__(self):
        if self._train:
            if self.sample_size is not None:
                return self.sample_size
            return self.train_instances
        return self.eval_instances

    def __getitem__(self, index):
        if self._train:
            return (self.word_train[index], None, index)
        return (self.word_eval[index], self.weights[index])

    def train(self):
        self._train = True

    def eval(self):
        self._train = False
