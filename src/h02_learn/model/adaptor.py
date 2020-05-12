import os
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from util.util import hacked_exp, write_torch_data, read_torch_data


class Adaptor:
    # pylint: disable=too-many-locals

    def __init__(self, alpha, beta, alphabet, state_folder, save_state=True):
        self.alpha = alpha
        self.beta = beta
        self.alphabet = alphabet
        self.saved_state_folder = state_folder
        self.save_state = save_state

        self.state = self.get_initial_state()

    def get_initial_state(self):
        state = {}
        # initialise mapping from table index to n.o. customers
        # int --> int
        state['customers_per_table'] = defaultdict(int)
        # initialise mapping from table indices to labels
        # int --> list(int)
        state['tables_with_word_label'] = defaultdict(set)
        # initialise mapping from customer id to table id
        # int --> int
        state['table_assignments'] = {}
        # this index doesn't have to be "accurate"
        # there may be gaps in the indices as some tables are removed
        # but we just want to make sure that every table index is unique
        state['max_table_index'] = -1
        # this is marked as the function K in the original paper
        state['total_tables'] = 0
        state['alpha'] = torch.Tensor([self.alpha])
        state['beta'] = torch.Tensor([self.beta])
        state['alphabet'] = self.alphabet
        return state

    def _sample_new_table_assignment(self, table_probs):
        probs, ids = zip(*table_probs)
        table_index = np.random.choice(ids, 1, p=probs)[0]
        if table_index < 0:
            # choose new table index
            # increment counter for total amount of tables
            self.state['total_tables'] += 1
            # increment table id counter
            self.state['max_table_index'] += 1
            return self.state['max_table_index']
        return table_index

    def calculate_cross_entropy(self, dataloader, generator):
        entropy = 0
        total_tokens = 0
        for x, y, weights, _ in tqdm(dataloader, total=len(dataloader), \
                                    desc='Calculating adaptor cross entropy', mininterval=.2):
            generator_logprobs = generator.get_word_log_probability(x, y)
            for i, log_prob in enumerate(generator_logprobs):
                # do not use the start of word index
                token_indices = x[i][1:]
                token = ''.join(self.state['alphabet'].idx2word(token_indices))
                word_logprob = self.get_token_logprobability(log_prob, token)
                entropy += -word_logprob * weights[i]
                total_tokens += weights[i]
        return (entropy / total_tokens).item()

    def get_token_logprobability(self, generator_logprob, token):
        """ The marginal probability of a token defined by marginalising over
            table assignments as defined by Goldwater et al. """
        i = self.state['dataset_length']
        tables_with_word_label = self.state['tables_with_word_label'][token]
        customers_in_tables_with_label = self.state['customers_in_tables_with_label'][token]
        adaptor_state = self.state['total_tables']*self.state['alpha'] + self.state['beta']
        if len(tables_with_word_label) == 0 and customers_in_tables_with_label == 0:
            # this takes care of rare words not encountered in training
            # their probabilities are too small to take away from log space
            return torch.log(adaptor_state) + generator_logprob - torch.log(i+self.state['beta'])
        generator_prob = torch.exp(generator_logprob)
        state1 = customers_in_tables_with_label - len(tables_with_word_label)*self.state['alpha']
        return torch.log(state1 + adaptor_state*generator_prob)-torch.log(i+self.state['beta'])

    def count_customers_in_tables_with_label(self, dataloader):
        c_in_tables_with_label = defaultdict(int)
        for x, _, _, _ in dataloader:
            for word_indices in x:
                word = ''.join(self.state['alphabet'].idx2word(word_indices[1:]))
                c_in_tables_with_label[word] = sum([self.state['customers_per_table'][table_id] \
                                        for table_id in self.state['tables_with_word_label'][word]])
        return c_in_tables_with_label

    @staticmethod
    def _normalise_table_probabilities(table_logprobs):
        exp_probs = hacked_exp([prob for prob, idd in table_logprobs])
        normaliser = sum(exp_probs)
        table_probs = [(prob/normaliser, table_logprobs[i][1]) \
                            for i, prob in enumerate(exp_probs)]
        return table_probs

    def _calculate_table_logprobs(self, token, token_logprob):
        table_logprobs = []
        # calculate probability of assigning to old table
        for table_id in self.state['tables_with_word_label'][token]:
            table_prob = torch.log(self.state['customers_per_table'][table_id]\
                                        - self.state['alpha'])
            table_logprobs.append((table_prob.item(), table_id))
        # calculate probability of assigning to new table
        new_table_logprob = torch.log(torch.Tensor([self.state['total_tables']*self.state['alpha']+\
                                                        self.state['beta']])) + token_logprob
        table_logprobs.append((new_table_logprob.item(), -1))
        return table_logprobs

    def fit(self, generator, dataloader):
        self.state['dataset_length'] = len(dataloader.dataset)

        for x, y, _, token_ids in tqdm(dataloader, total=len(dataloader), \
                                    desc='Fitting adaptor', mininterval=.2):
            tokens_logprobs = generator.get_word_log_probability(x, y)
            # iterate through tokens in batch:
            for i, token_logprob in enumerate(tokens_logprobs):
                token_id = token_ids[i].item()
                token_indices = x[i][1:]
                token = ''.join(self.state['alphabet'].idx2word(token_indices))
                if token_id in self.state['table_assignments']:
                    token_table_id = self.state['table_assignments'][token_id]
                    # remove customer from table
                    self.state['customers_per_table'][token_table_id] -= 1
                    # if table is empty then don't associate with word anymore
                    if self.state['customers_per_table'][token_table_id] == 0:
                        self.state['tables_with_word_label'][token].remove(token_table_id)
                        self.state['total_tables'] -= 1
                table_logprobs = self._calculate_table_logprobs(token, token_logprob)
                table_probs = self._normalise_table_probabilities(table_logprobs)
                assigned_table_id = self._sample_new_table_assignment(table_probs)
                # put customer to new table
                self.state['customers_per_table'][assigned_table_id] += 1
                # store info about amount of labels
                self.state['tables_with_word_label'][token].add(assigned_table_id)
                self.state['table_assignments'][token_id] = assigned_table_id

        if self.save_state:
            print('Saving adaptor state to', self.saved_state_folder)
            self.save_fitted_state(dataloader)

        print('Done fitting the adaptor')
        return self.state['tables_with_word_label']

    def save_fitted_state(self, dataloader, state_folder=None):
        customers_in_tables_with_label = self.count_customers_in_tables_with_label(dataloader)
        self.state['customers_in_tables_with_label'] = customers_in_tables_with_label

        if state_folder is None:
            state_folder = self.saved_state_folder
        adaptor_state_file = self.get_state_file(state_folder)
        write_torch_data(adaptor_state_file, self.get_checkpoint())

    @classmethod
    def load(cls, state_folder):
        state_file = cls.get_state_file(state_folder)
        print('Loading fitted adaptor from', state_file)
        checkpoint = read_torch_data(state_file)
        adaptor = cls(**checkpoint['kwargs'], state_folder=state_folder)
        adaptor.set_state(checkpoint['state'])
        return adaptor

    def set_state(self, state):
        self.state = state

    def get_checkpoint(self):
        return {
            'kwargs': self.get_args(),
            'state': self.state
        }

    def get_args(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'alphabet': self.alphabet,
            'save_state': self.save_state,
        }

    @staticmethod
    def get_state_file(saved_state_folder):
        return os.path.join(saved_state_folder, 'adaptor_state')
