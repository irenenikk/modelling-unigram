import sys
import logging
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from h01_data.alphabet import Alphabet
from util import argparser
from util import util


def get_args():
    argparser.add_argument(
        "--wikipedia-tokenized-file", type=str,
        help="The file in which wikipedia tokenized results should be")
    argparser.add_argument(
        "--data-file", type=str,
        help="The file in which processed data should be saved")
    argparser.add_argument(
        "--n-folds", type=int, default=10,
        help="Number of folds to split data")

    return argparser.parse_args()


def count_sentences(fname):
    count = 0
    with open(fname, 'r') as f:
        for _ in f.readlines():
            count += 1
    return count


def get_fold_splits(n_sentences, n_folds):
    splits = np.arange(n_sentences)
    np.random.shuffle(splits)
    splits = np.array_split(splits, n_folds)
    splits = {x: i for i, fold in enumerate(splits) for x in fold}
    return splits


def process_line(line, word_info, alphabet):
    for word in line.strip().split(' '):
        alphabet.add_word(word)

        if word in word_info:
            word_info[word]['count'] += 1
        else:
            word_info[word] = {
                'count': 1,
                'idx': alphabet.word2idx(word)
            }


def process_data(src_fname, n_folds, splits, alphabet):
    folds = [{} for _ in range(n_folds)]
    with open(src_fname, 'r') as f:
        for i, line in tqdm(enumerate(f.readlines()), desc='Processing wiki data',
                            total=len(splits)):
            fold = splits[i]
            process_line(line, folds[fold], alphabet)

    return folds


def count_tokens(folds):
    return [sum([x['count'] for x in word_info.values()]) for word_info in folds]


def count_types(folds):
    return [len(word_info) for word_info in folds]


def process(src_fname, tgt_fname, n_folds):
    # spacy_tokenizer = load_spacy(spacy_option)
    n_sentences = count_sentences(src_fname)
    splits = get_fold_splits(n_sentences, n_folds)
    alphabet = Alphabet()

    folds = process_data(src_fname, n_folds, splits, alphabet)
    n_tokens = count_tokens(folds)
    n_types = count_types(folds)
    util.write_data(tgt_fname, (folds, alphabet, n_tokens))

    print('# unique chars:', len(alphabet))
    print('# tokens per fold:', n_tokens)
    print('# types per fold:', n_types)


def main():
    args = get_args()
    logging.info(args)

    process(args.wikipedia_tokenized_file, args.data_file, args.n_folds)


if __name__ == '__main__':
    main()
