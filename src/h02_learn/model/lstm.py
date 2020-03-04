import torch.nn as nn
import torch.nn.functional as F

from util import constants
from .base import BaseLM


class LstmLM(BaseLM):
    # pylint: disable=arguments-differ
    name = 'lstm'

    def __init__(self, alphabet_size, embedding_size, hidden_size,
                 nlayers, dropout, ignore_index=None):
        super().__init__(alphabet_size, embedding_size, hidden_size,
                         nlayers, dropout)

        self.embedding = nn.Embedding(alphabet_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, nlayers,
                            dropout=(dropout if nlayers > 1 else 0),
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.out = nn.Linear(embedding_size, alphabet_size)

        # Tie weights
        self.out.weight = self.embedding.weight
        self.ignore_index = ignore_index

    def forward(self, x):
        x_emb = self.dropout(self.embedding(x))

        c_t, _ = self.lstm(x_emb)
        c_t = self.dropout(c_t).contiguous()

        hidden = F.relu(self.linear(c_t))
        logits = self.out(hidden)
        return logits

    def get_word_log_probability(self, x, y):
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index,\
                                            reduction='none').to(device=constants.device)
        logits = self(x)
        logprobs = criterion(logits.reshape(-1, logits.shape[-1]),\
                                y.reshape(-1)).reshape_as(y).sum(-1)
        return -logprobs
