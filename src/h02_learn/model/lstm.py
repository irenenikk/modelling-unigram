import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def get_word_probability(self, x, y):
        # get model predictions
        logits = self(x)
        # probs is shaped (batch_size, word_length, n_characters)
        probs = F.softmax(logits, dim=2)
        mask = np.zeros(y.shape, dtype=bool)
        mask[(y == self.ignore_index)] = True
        expanded_mask = np.broadcast_to(mask.T, (1, y.shape[0], y.shape[1], probs.shape[2]))
        probs[expanded_mask] = 1
        # calculate the total probability from predictions
        # probs.gather is shaped (batch_size, word_length)
        # probs.gather.prod is shaped (batch_size)
        log_probs = torch.log(torch.gather(probs, 2, y.unsqueeze(2))).sum(-2)
        return log_probs

    def get_logprobs(self, x, y):
        # this is an unused method for now
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduce='none') \
        .to(device=constants.device)
        logprobs = criterion(logits, y)
