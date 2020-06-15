import os
import sys
import math
import numpy as np
import torch
import scipy.stats as stats

sys.path.append('./src/')
from h02_learn.dataset import load_data, get_data_loaders_with_folds
from h02_learn.train_generator import load_generator
from h02_learn.model.adaptor import Adaptor
from util import util
from util.argparser import get_argparser, parse_args, add_data_args, add_generator_args
from util import constants

def get_args():
    argparser = get_argparser()
    # adaptor
    argparser.add_argument('--two-stage-state-folder', type=str, required=True)
    argparser.add_argument('--results-file', type=str, required=True)
    add_data_args(argparser)
    add_generator_args(argparser)
    args = parse_args(argparser)
    return args

# pylint: disable=too-many-arguments
def save_results(model, atural_code_avg, permuted_natural_avg, two_stage_avg,\
                    natural_correlation, permuted_correlation, two_stage_correlation,\
                    alphabet_size, sentences, results_fname, test):
    print('Saving to', results_fname)
    results = []
    file_size = os.path.getsize(results_fname) if os.path.exists(results_fname) else 0
    if file_size == 0:
        results = [['model', 'natural_code_avg', 'permuted_natural_code_avg',\
                    'two_stage_code_avg', 'natural_correlation', 'permuted_correlation',\
                    'two_stage_correlation', 'alphabet_size', 'sentences', 'test']]
    results += [[model, atural_code_avg, permuted_natural_avg, two_stage_avg,\
                natural_correlation, permuted_correlation, two_stage_correlation,\
                alphabet_size, sentences, test]]
    util.write_csv(results_fname, results)


def average_sentence_length(sentences, type_lengths):
    lengths = []
    for sentence in sentences:
        word_lengths = sum([type_lengths[word] for word in sentence])
        whitespace_length = len(sentence) - 1
        lengths.append(word_lengths + whitespace_length)
    return np.mean(lengths), np.std(lengths)


def calculate_word_lengths_under_code(sentences, type_logprobs, alphabet_size):
    type_lengths = {}
    for sentence in sentences:
        for word in sentence:
            if word in type_lengths:
                continue
            original_logprob = type_logprobs[word]
            new_logprob = original_logprob/np.log(alphabet_size)
            code_length = math.ceil(-new_logprob)
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
    generator.eval()
    with torch.no_grad():
        prob = generator.get_word_log_probability(x_batch, y_batch)
    return prob


def calculate_word_logprobability(word, adaptor, generator, alphabet):
    generator_logprob = get_generator_word_probability(generator, word, alphabet)
    word_logprob = adaptor.get_token_logprobability(generator_logprob, word)
    return word_logprob.item()


def calculate_two_stage_type_probs(sentences, adaptor, generator, alphabet):
    type_logprobs = {}
    for sentence in sentences:
        for word in sentence:
            if word in type_logprobs:
                continue
            logprob = calculate_word_logprobability(word, adaptor, generator, alphabet)
            type_logprobs[word] = logprob
    return type_logprobs


def calculate_natural_code_average(sentences):
    natural_code_lengths = calculate_word_lengths(sentences)
    return average_sentence_length(sentences, natural_code_lengths)


def calculate_permuted_code_lengths(sentences):
    natural_code_lengths = calculate_word_lengths(sentences)
    permuted_code_lengths = util.permute_dict(natural_code_lengths)
    return permuted_code_lengths


def calculate_random_code_average(sentences):
    permuted_code_lengths = calculate_permuted_code_lengths(sentences)
    return average_sentence_length(sentences, permuted_code_lengths)


def calculate_two_stage_code_lengths(sentences, adaptor, generator, alphabet):
    type_logprobs = calculate_two_stage_type_probs(sentences, adaptor, generator, alphabet)
    type_code_lengths = calculate_word_lengths_under_code(sentences, type_logprobs, len(alphabet))
    return type_code_lengths


def calculate_two_stage_code_average(sentences, adaptor, generator, alphabet):
    type_code_lengths = calculate_two_stage_code_lengths(sentences, adaptor, generator, alphabet)
    return average_sentence_length(sentences, type_code_lengths)


def correlation(type_lengths, type_freqs):
    lengths = [type_lengths[k] for k in type_freqs.keys()]
    freqs = [type_freqs[k] for k in type_freqs.keys()]
    pearson = stats.pearsonr(lengths, freqs)
    spearman = stats.spearmanr(lengths, freqs)
    return pearson[0], spearman[0]


def calculate_all_correlatios(sentences, adaptor, generator, alphabet, type_freqs):
    natural_code_lengths = calculate_word_lengths(sentences)
    permuted_code_lengths = calculate_permuted_code_lengths(sentences)
    two_stage_code_lengths = \
        calculate_two_stage_code_lengths(sentences, adaptor, generator, alphabet)
    natural_correlation = correlation(natural_code_lengths, type_freqs)
    permuted_correlation = correlation(permuted_code_lengths, type_freqs)
    two_stage_correlation = correlation(two_stage_code_lengths, type_freqs)
    return natural_correlation, permuted_correlation, two_stage_correlation


def run_experiments(sentences, generator, adaptor, alphabet, data_loader, args, test):
    natural_avg = \
        calculate_natural_code_average(sentences)
    natural_perm_avg = \
        calculate_random_code_average(sentences)
    two_stage_avg = \
        calculate_two_stage_code_average(sentences, adaptor, generator, alphabet)
    natural_corr, permuted_corr, two_stage_corr = \
        calculate_all_correlatios(sentences, adaptor, generator, alphabet,
                                  dict(data_loader.dataset.word_freqs))
    print('Natural code average sentence length:', natural_avg,
          '(test=', test, ')')
    print('Natural code average sentence length with permuted lengths:', natural_perm_avg,
          '(test=', test, ')')
    print('Two-stage code average sentence length:', two_stage_avg,
          '(test=', test, ')')
    save_results(args.two_stage_state_folder, natural_avg, natural_perm_avg,\
                 two_stage_avg, natural_corr, permuted_corr, two_stage_corr,
                 len(alphabet), len(sentences), args.results_file, test=test)


def main():
    args = get_args()
    folds = util.get_folds()

    data = load_data(args.data_file)
    _, sentence_data, alphabet, _ = data
    dev_sentences = sentence_data[folds[1][0]]
    test_sentences = sentence_data[folds[2][0]]
    _, dev_loader, test_loader, _ = get_data_loaders_with_folds('tokens', args.data_file, folds,\
                                                             args.batch_size)

    generator = load_generator(args.two_stage_state_folder)
    adaptor = Adaptor.load(args.two_stage_state_folder)

    run_experiments(dev_sentences, generator, adaptor, alphabet, dev_loader, args, test=False)
    run_experiments(test_sentences, generator, adaptor, alphabet, test_loader, args, test=True)

if __name__ == '__main__':
    main()
