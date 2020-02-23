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
        self.token_dataloader = dataloader
        self.alphabet = alphabet
        print('Token data length', len(self.token_dataloader))

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

    def calculate_cross_entropy(dataloader, generator):
        entropy = 0
        for x, y, weights in dataloader:
            # TODO: optimise
            for i, word_indices in enumerate(x):
                word = ''.join(self.alphabet.idx2word(word_indices))
                word_logprob = self.get_token_probability(generator, word)
                entropy += -word_logprob * weights[i]
        return entropy / len(dataloader.dataset)


    def get_token_probability(self, generator, token):
        generator_prob = self.get_generator_word_probability(generator, token)
        customers_in_tables_with_label = sum([self.customers_per_table[table] for table_id in self.tables_with_word_label[token]])
        i = len(self.token_dataloader)
        return (customers_in_tables_with_label - len(self.tables_with_word_label[token])*self.a \
                + (self.total_tables*self.a + self.b)*generator_prob)/(i+self.b)

    def get_generator_word_probability(self, generator, batch):
        word_char_indices = [self.token_dataloader.dataset.get_word_idx(word) for word in batch]
        x = word_char_indices[:-1]
        y = word_char_indices[1:]
        x_batch = torch.LongTensor([x])
        y_batch = torch.LongTensor([y])
        log_probs = generator.get_word_probability(x_batch, y_batch)
        #print('log probs', log_probs)
        return log_probs

    def fit(self, generator):
        for tokens_indices, target_indices, token_ids in tqdm(self.token_dataloader, total=len(self.token_dataloader), desc='Fitting adaptor', mininterval=.2):
            tokens_logprobs = generator.get_word_probability(tokens_indices, target_indices)
            # iterate through tokens in batch:
            for i in range(len(tokens_logprobs)):
                token_id = token_ids[i].item()
                token_indices = tokens_indices[i][1:]
                token = ''.join(self.alphabet.idx2word(token_indices))
                print('token id', token_id)
                print('token', token)
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
                    print('self.customers_per_table', table_id, ':', self.customers_per_table[table_id])
                    table_prob = np.log(self.customers_per_table[table_id] - self.a)
                    table_probs.append((table_prob, table_id))
                # calculate probability of assigning to new table
                #print('token logprob', token_logprob)
                #print('self.total_tables + self.b', self.total_tables + self.b)
                new_table_prob = torch.log(torch.Tensor([self.total_tables*self.a + self.b])) + tokens_logprobs[i]
                #print('new table probability', new_table_prob)
                table_probs.append((new_table_prob.squeeze().detach().numpy(), -1))
                # normalise to probabilities before sampling
                exp_probs = hacked_exp([prob for prob, idd in table_probs])
                normaliser = sum(exp_probs)
                table_probs = [(prob/normaliser, table_probs[i][1]) for i, prob in enumerate(exp_probs)]
                print('table probabilities', table_probs)
                assigned_table_id = self._sample_new_table_assignment(table_probs)
                # put customer to new table
                self.customers_per_table[assigned_table_id] += 1
                # store info about amount of labels
                self.tables_with_word_label[token].add(assigned_table_id)
                #print('tables with word label', self.tables_with_word_label[token])
                self.table_assignments[token_id] = assigned_table_id
        print('Done fitting the adaptor')
