import os

from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from util.util import write_data, read_data, create_int_defaultdict

class Adaptor:
    # pylint: disable=too-many-locals

    def __init__(self, state, state_folder, save_state=True):
        self.state = state
        self.saved_state_folder = state_folder
        self.save_state = save_state

    @staticmethod
    def get_initial_state(alpha, beta, alphabet):
        state = {}
        # optimised representation:
        # { word: { customer amount: table amount } }
        state['seating_histogram_per_word'] = defaultdict(create_int_defaultdict)
        state['customers_in_tables_with_label'] = defaultdict(int)
        # initialise mapping from table indices to labels
        # { word : table amount }
        state['tables_with_word_label'] = defaultdict(int)
        state['assigned_to_table'] = defaultdict(bool)
        state['total_tables'] = 0
        state['alpha'] = torch.Tensor([alpha])
        state['beta'] = torch.Tensor([beta])
        state['alphabet'] = alphabet
        return state

    @staticmethod
    def get_state_file(saved_state_folder):
        return os.path.join(saved_state_folder, 'adaptor_state')

    @classmethod
    def load(cls, state_folder):
        state_file = cls.get_state_file(state_folder)
        print('Loading fitted adaptor from', state_file)
        state = read_data(state_file)
        adaptor = cls(state, state_folder)
        return adaptor

    def set_state(self, state):
        self.state = state

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
                word_logprob = self.get_token_probability(log_prob, token)
                entropy += -word_logprob * weights[i]
                total_tokens += weights[i]
        return (entropy / total_tokens).item()

    def get_token_probability(self, generator_logprob, token):
        """ The marginal probability of a token defined by marginalising over
            table assignments as defined by Goldwater et al. """
        i = self.state['dataset_length']
        tables_with_word_label = self.state['tables_with_word_label'][token]
        customers_in_tables_with_label = self.state['customers_in_tables_with_label'][token]
        if tables_with_word_label == 0 and customers_in_tables_with_label == 0:
            # this takes care of rare words not encountered in training
            # their probabilities are too small to take away from log space
            adaptor_state = self.state['total_tables']*self.state['alpha'] + self.state['beta']
            return torch.log(adaptor_state) + generator_logprob - torch.log(i+self.state['beta'])
        generator_prob = torch.exp(generator_logprob)
        state1 = customers_in_tables_with_label - tables_with_word_label*self.state['alpha']
        state2 = self.state['total_tables']*self.state['alpha'] + self.state['beta']
        res = torch.log(state1 + state2*generator_prob)-torch.log(i+self.state['beta'])
        return res

    def save_fitted_state(self, state_folder=None):
        saved_state_folder = state_folder
        if state_folder is None:
            saved_state_folder = self.saved_state_folder
        adaptor_state_file = self.get_state_file(saved_state_folder)
        write_data(adaptor_state_file, self.state)

    def customer_enters(self, word, word_logprob):
        customers_in_tables = self.state['customers_in_tables_with_label'][word]
        tables_with_word_label = self.state['tables_with_word_label'][word]
        old_table_prob = (customers_in_tables - self.state['alpha']*tables_with_word_label).item()
        logstate = torch.log(self.state['beta'] + self.state['total_tables']*self.state['alpha'])
        new_table_prob = torch.exp(logstate + word_logprob).item()
        random_draw = np.random.uniform(0, old_table_prob + new_table_prob)
        if random_draw < new_table_prob or customers_in_tables == 0:
            # put into a new table
            self.state['seating_histogram_per_word'][word][1] += 1
            self.state['total_tables'] += 1
            self.state['tables_with_word_label'][word] += 1
        else:
            # put into an existing table based on histogram
            random_draw = np.random.uniform(0, customers_in_tables)
            for customer_amount in self.state['seating_histogram_per_word'][word]:
                random_draw -= customer_amount * \
                                self.state['seating_histogram_per_word'][word][customer_amount]
                if random_draw <= 0:
                    self.state['seating_histogram_per_word'][word][customer_amount+1] += 1
                    self.state['seating_histogram_per_word'][word][customer_amount] -= 1
                    break
        self.state['customers_in_tables_with_label'][word] += 1

    def customer_leaves(self, word):
        customers_in_tables = self.state['customers_in_tables_with_label'][word]
        random_draw = np.random.uniform(0, customers_in_tables)
        for customer_amount in self.state['seating_histogram_per_word'][word]:
            random_draw -= customer_amount * \
                            self.state['seating_histogram_per_word'][word][customer_amount]
            if random_draw <= 0:
                self.state['seating_histogram_per_word'][word][customer_amount] -= 1
                self.state['total_tables'] -= 1
                self.state['tables_with_word_label'][word] -= 1
                if customer_amount > 1:
                    # in case the table is not empty after the customer leaves
                    self.state['seating_histogram_per_word'][word][customer_amount-1] += 1
                    self.state['total_tables'] += 1
                    self.state['tables_with_word_label'][word] += 1
                break
        self.state['customers_in_tables_with_label'][word] -= 1


    def fit(self, generator, dataloader):
        self.state['dataset_length'] = len(dataloader)
        for x, y, _, token_ids in tqdm(dataloader, total=len(dataloader), \
                                    desc='Fitting adaptor', mininterval=.2):
            tokens_logprobs = generator.get_word_log_probability(x, y)
            # iterate through tokens in batch:
            for i, token_logprob in enumerate(tokens_logprobs):
                token_id = token_ids[i].item()
                token_indices = x[i][1:]
                token = ''.join(self.state['alphabet'].idx2word(token_indices))
                if self.state['assigned_to_table'][token_id]:
                    self.customer_leaves(token)
                self.customer_enters(token, token_logprob)
                self.state['assigned_to_table'][token_id] = True
        if self.save_state:
            print('Saving adaptor state to', self.saved_state_folder)
            self.save_fitted_state()
        print('Done fitting the adaptor')
        return self.state['tables_with_word_label']
