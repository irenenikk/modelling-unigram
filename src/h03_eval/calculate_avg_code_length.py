import os
import sys
import math
import numpy as np
import torch

sys.path.append('./src/')
from h02_learn.dataset import load_data
from h02_learn.train_generator import load_generator
from h02_learn.model.adaptor import Adaptor
from util import util
from util.argparser import get_argparser, parse_args, add_data_args, add_generator_args
from util import constants

def get_args():
    argparser = get_argparser()
    # adaptor
    argparser.add_argument('--two-stage-state-folder', type=str, required=True)
    add_data_args(argparser)
    add_generator_args(argparser)
    args = parse_args(argparser)
    return args


def save_pitman_yor_results(model, dataset, alpha, beta, train_loss,\
                            dev_loss, test_loss, test_size, results_fname):
    print('Saving to', results_fname)
    results = []
    file_size = os.path.getsize(results_fname) if os.path.exists(results_fname) else 0
    if file_size == 0:
        results = [['dataset', 'alphabet_size', 'embedding_size', 'hidden_size', 'nlayers', 'dropout_p',\
                     'alpha', 'beta', 'train_loss', 'dev_loss', 'test_loss', 'test_size']]
    results += [[dataset, model.alphabet_size, model.embedding_size, model.hidden_size, model.nlayers,\
                model.dropout_p, alpha, beta, train_loss, dev_loss, test_loss, test_size]]
    util.write_csv(results_fname, results)


def average_sentence_length(sentences, type_lengths):
    lengths = [type_lengths[word] for sentence in sentences for word in sentence]
    return np.mean(lengths), np.std(lengths)


def calculate_word_lengths_under_code(sentences, type_probs, alphabet_size):
    type_lengths = {}
    for sentence in sentences:
        sentence_length = 0
        for word in sentence:
            if word in type_lengths:
                continue
            code_length = math.ceil(-math.log(type_probs[word], alphabet_size))
            type_lengths[word] = code_length
    return type_lengths


def calculate_word_lengths(sentences):
    lengths = {}
    for sentence in sentences:
        for word in sentence:
            lengths[word] = len(word)
    return lengths


def get_word_idx(word, alphabet):
    return [alphabet.char2idx('SOS')] + \
            alphabet.word2idx(word) + \
            [alphabet.char2idx('EOS')]


def get_generator_word_probability(generator, word, alphabet):
    word_char_indices = get_word_idx(word, alphabet)
    x = word_char_indices[:-1]
    y = word_char_indices[1:]
    x_batch = torch.LongTensor([x]).to(device=constants.device)
    y_batch = torch.LongTensor([y]).to(device=constants.device)
    return generator.get_word_log_probability(x_batch, y_batch)


def calculate_word_probability(word, adaptor, generator, alphabet):
    generator_logprob = get_generator_word_probability(generator, word, alphabet)
    word_logprob = adaptor.get_token_logprobability(generator_logprob, word)
    return np.exp(word_logprob)


def calculate_two_stage_type_probs(sentences, adaptor, generator, alphabet):
    type_probs = {}
    for sentence in sentences:
        for word in sentence:
            if word in type_probs:
                continue
            prob = calculate_word_probability(word, adaptor, generator, alphabet)
            type_probs[word] = prob
    return type_probs


def calculate_natural_code_average(sentences):
    natural_code_lengths = calculate_word_lengths(sentences)
    return average_sentence_length(sentences, natural_code_lengths)


def calculate_random_code_average(sentences):
    natural_code_lengths = calculate_word_lengths(sentences)
    permuted_code_lengths = util.permute_dict(natural_code_lengths)
    return average_sentence_length(sentences, permuted_code_lengths)


def calculate_two_stage_code_average(sentences, adaptor, generator, alphabet):
    type_probs = calculate_two_stage_type_probs(sentences, adaptor, generator, alphabet)
    type_code_lengths = calculate_word_lengths_under_code(sentences, type_probs, len(alphabet))
    return average_sentence_length(sentences, type_code_lengths)


def main():
    # pylint: disable=all
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    data = load_data(args.data_file)
    _, sentence_data, alphabet, n_tokens = data
    dev_sentences = sentence_data[folds[1][0]]
    test_sentences = sentence_data[folds[2][0]]

    generator = load_generator(args.two_stage_state_folder)
    generator.eval()
    adaptor = Adaptor.load(args.two_stage_state_folder)

    natural_code_average = calculate_natural_code_average(dev_sentences)
    natural_permuted_code_average = calculate_random_code_average(dev_sentences)
    two_stage_code_average = calculate_two_stage_code_average(dev_sentences, adaptor, generator, alphabet)

    print('Natural code average sentence length', natural_code_average)
    print('Natural code average sentence length with permuted lengths', natural_permuted_code_average)
    print('Two-stage code average sentence length', two_stage_code_average)

    save_pitman_yor_results(generator, args.dataset, alpha, beta, train_loss, dev_loss, test_loss,\
                            len(testloader.dataset), args.adaptor_results_file)


if __name__ == '__main__':
    main()
