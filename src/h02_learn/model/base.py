import copy
import torch
import torch.nn as nn

from util import constants


class BaseLM(nn.Module):
    name = 'base'

    def __init__(self, alphabet_size, embedding_size, hidden_size,
                 nlayers, dropout):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout
        self.alphabet_size = alphabet_size

        self.best_state_dict = None

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)

    def save(self, path):
        fname = self.get_name(path)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    def get_args(self):
        return {
            'nlayers': self.nlayers,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'dropout': self.dropout_p,
            'alphabet_size': self.alphabet_size,
        }

    @classmethod
    def load(cls, path):
        checkpoints = cls.load_checkpoint(path)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    @classmethod
    def load_checkpoint(cls, path):
        fname = cls.get_name(path)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path):
        return '%s/model.tch' % (path)
