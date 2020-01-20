import sys
import logging
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from util import argparser
from util import util


def get_args():
    argparser.add_argument(
        "--wikipedia-tokenized-file", type=str,
        help="The file in which wikipedia tokenized results should be")
    argparser.add_argument(
        "--processed-data-file", type=str,
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


def process_line(line, word_counts):
    for word in line.strip().split(' '):
        word_counts[word] = word_counts.get(word, 0) + 1


def process_data(src_fname, n_folds, splits):
    fold_counts = [{} for _ in range(n_folds)]
    with open(src_fname, 'r') as f:
        for i, line in tqdm(enumerate(f.readlines()), desc='Processing wiki data',
                            total=len(splits)):
            fold = splits[i]
            process_line(line, fold_counts[fold])

    return fold_counts


def count_tokens(fold_counts):
    return [sum([x for x in word_counts.values()]) for word_counts in fold_counts]


def count_types(fold_counts):
    return [len(word_counts) for word_counts in fold_counts]


def process(src_fname, tgt_fname, n_folds):
    # spacy_tokenizer = load_spacy(spacy_option)
    n_sentences = count_sentences(src_fname)
    splits = get_fold_splits(n_sentences, n_folds)

    fold_counts = process_data(src_fname, n_folds, splits)
    n_tokens = count_tokens(fold_counts)
    n_types = count_types(fold_counts)
    util.write_data(tgt_fname, (fold_counts, n_tokens))

    print('# tokens per fold:', n_tokens)
    print('# types per fold:', n_types)


def main():
    args = get_args()
    logging.info(args)

    process(args.wikipedia_tokenized_file, args.processed_data_file, args.n_folds)


if __name__ == '__main__':
    main()
