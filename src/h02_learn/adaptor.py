import numpy as np
import random
import torch
from collections import defaultdict
from util.util import hacked_exp
from tqdm import tqdm

class Adaptor:
    def __init__(self, a, b, alphabet, dataloader):
        # initialise mapping from table index to n.o. customers (c)
        # int --> int
        self.customers_per_table = defaultdict(int)
        # initialise mapping from table indices to labels (t)
        # int --> list(int)
        self.tables_with_word_label = defaultdict(set)
        # initialise mapping from customer id to table id (z)
        # int --> int
        self.table_assignments = {}
        # this index doesn't have to be "accurate"
        # there may be gaps in the indices as some tables are removed
        # but we just want to make sure that every table index is unique
        self.max_table_index = -1
        # this is marked as the function K in the original paper
        self.total_tables = 0
        self.a = a
        self.b = b
        self.tokens = dataloader
        self.alphabet = alphabet
        print('Token data length', len(self.tokens))

    def _sample_new_table_assignment(self, table_probs):
        ids = [idd for prob, idd in table_probs]
        probs = [prob for prob, idd in table_probs]
        table_index = np.random.choice(ids, 1, p=probs)[0]
        if table_index < 0:
            # choose new table index
            # increment counter for total amount of tables
            self.total_tables += 1
            # increment table id counter
            self.max_table_index += 1
            return self.max_table_index
        return table_index

    def get_token_probability(self, generator, token):
        generator_prob = self.generator_word_probability(generator, token)
        customers_in_tables_with_label = sum([self.customers_per_table[table] for table_id in self.tables_with_word_label[token]])
        i = len(self.tokens)
        return (customers_in_tables_with_label - self.tables_with_word_label[token]*self.a \
                + (self.total_tables*self.a + self.b)*generator_prob)/(i+b)

    def get_generator_word_probability(self, generator, word):
        word_char_indices = self.tokens.dataset.get_word_idx(word)
        x = word_char_indices[:-1]
        y = word_char_indices[1:]
        x_batch = torch.LongTensor([x])
        y_batch = torch.LongTensor([y])
        log_probs = generator.get_word_probability(x_batch, y_batch)
        #print('log probs', log_probs)
        return log_probs

    def fit(self, generator):
        #for token_indices, _, token_id in tqdm(self.tokens, total=len(self.tokens), desc='Fitting adaptor', mininterval=.2):
        for token_indices, _, token_id in self.tokens:
            token_id = token_id.item()
            token_indices = token_indices[0][1:]
            token = ''.join(self.alphabet.idx2word(token_indices))
            #print('token id', token_id)
            #print('token', token)
            if token_id in self.table_assignments:
                token_table_id = self.table_assignments[token_id]
                # remove customer from table
                self.customers_per_table[token_table_id] -= 1
                # if table is empty then don't associate with word anymore
                if self.customers_per_table[token_table_id] == 0:
                    #print('Table removed since zero customers')
                    self.tables_with_word_label[token].remove(self.table_assignments[token_id])
                    self.total_tables -= 1
            table_probs = []
            # calculate probability of assigning to old table
            for table_id in self.tables_with_word_label[token]:
                #print('self.customers_per_table', table_id, ':', self.customers_per_table[table_id])
                table_prob = np.log(self.customers_per_table[table_id] - self.a)
                table_probs.append((table_prob, table_id))
            # calculate probability of assigning to new table
            token_logprob = self.get_generator_word_probability(generator, token)
            #print('token logprob', token_logprob)
            #print('self.total_tables + self.b', self.total_tables + self.b)
            new_table_prob = torch.log(torch.Tensor([self.total_tables*self.a + self.b])) + token_logprob
            #print('new table probability', new_table_prob)
            table_probs.append((new_table_prob.squeeze().detach().numpy(), -1))
            # normalise to probabilities before sampling
            exp_probs = hacked_exp([prob for prob, idd in table_probs])
            normaliser = sum(exp_probs)
            table_probs = [(prob/normaliser, table_probs[i][1]) for i, prob in enumerate(exp_probs)]
            #print('table probabilities', table_probs)
            assigned_table_id = self._sample_new_table_assignment(table_probs)
            # put customer to new table
            self.customers_per_table[assigned_table_id] += 1
            # store info about amount of labels
            self.tables_with_word_label[token].add(assigned_table_id)
            #print('tables with word label', self.tables_with_word_label[token])
            self.table_assignments[token_id] = assigned_table_id
