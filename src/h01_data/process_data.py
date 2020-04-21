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
        "--n-folds", type=int, default=10,
        help="Number of folds to split data")
    argparser.add_argument(
        "--max-sentences", type=int, default=4000,
        help="Maximum number of sentences used")
    argparser.add_argument(
        "--sample-size", type=int,
        help="Sample size")
    return argparser.parse_args()


def count_sentences(fname):
    count = 0
    with open(fname, 'r') as f:
        for _ in f.readlines():
            count += 1
    return count


def get_fold_splits(n_sentences, n_folds, max_sentences=None):
    splits = np.arange(n_sentences)
    np.random.shuffle(splits)
    if max_sentences is not None:
        splits = splits[:max_sentences]
    splits = np.array_split(splits, n_folds)
    splits = {x: i for i, fold in enumerate(splits) for x in fold}
    return splits


def process_line(line, word_info, alphabet):
    split_line = line.strip().split(' ')
    for word in split_line:
        # exclude words that contain non-letters
        if not word.isalpha():
            continue
        alphabet.add_word(word)

        if word in word_info:
            word_info[word]['count'] += 1
        else:
            word_info[word] = {
                'count': 1,
                'idx': alphabet.word2idx(word)
            }
    return len(split_line)


def process_data(src_fname, n_folds, splits, alphabet, sample_size):
    total_words = 0
    folds = [{} for _ in range(n_folds)]
    with open(src_fname, 'r') as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        for i, line in tqdm(enumerate(lines), desc='Processing wiki data',
                            total=len(splits)):
            if i in splits:
                fold = splits[i]
                no_words = process_line(line, folds[fold], alphabet)
                total_words += no_words
                if total_words >= sample_size:
                    break
    return folds


def count_tokens(folds):
    return [sum([x['count'] for x in word_info.values()]) for word_info in folds]


def count_types(folds):
    return [len(word_info) for word_info in folds]


def process(src_fname, tgt_fname, n_folds, sample_size, max_sentences=None):
    # spacy_tokenizer = load_spacy(spacy_option)
    n_sentences = count_sentences(src_fname)
    splits = get_fold_splits(n_sentences, n_folds, max_sentences=max_sentences)
    alphabet = Alphabet()

    folds = process_data(src_fname, n_folds, splits, alphabet, sample_size)
    n_tokens = count_tokens(folds)
    n_types = count_types(folds)
    util.write_data(tgt_fname, (folds, alphabet, n_tokens))

    print('# unique chars:', len(alphabet))
    print('# tokens per fold:', n_tokens)
    print('# types per fold:', n_types)


def main():
    args = get_args()
    logging.info(args)

    process(args.wikipedia_tokenized_file, args.data_file, args.n_folds, args.sample_size, args.max_sentences)


if __name__ == '__main__':
    main()
