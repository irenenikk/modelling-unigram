import sys
import logging
import string
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from h01_data.alphabet import Alphabet
from util.argparser import get_argparser, parse_args, add_data_args
from util import util


def get_args():
    argparser = get_argparser()
    argparser.add_argument(
        "--wikipedia-tokenized-file", type=str,
        help="The file in which wikipedia tokenized results should be")
    argparser.add_argument(
        "--n-folds", type=int, default=10,
        help="Number of folds to split data")
    argparser.add_argument(
        "--max-sentences", type=int, default=4000,
        help="Maximum number of sentences used")
    add_data_args(argparser)
    return parse_args(argparser)


def count_sentences(fname):
    count = 0
    with open(fname, 'r') as f:
        for _ in f:
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

def process_line(line, word_info, sentence_list, alphabet):
    # remove punctuation
    line = line.translate(str.maketrans('', '', string.punctuation))
    sentence = [word.lower() for word in list(filter(None, line.strip().split(' ')))]
    # only accept words without extra symbols
    is_allowed = all([all([char in string.ascii_lowercase
                           for char in word.lower()])
                      for word in sentence])
    if not is_allowed:
        return
    sentence_list.append(sentence)
    for word in sentence:
        # exclude words that contain non-letters
        word = word.lower()
        alphabet.add_word(word)

        if word in word_info:
            word_info[word]['count'] += 1
        else:
            word_info[word] = {
                'count': 1,
                'idx': alphabet.word2idx(word)
            }


def process_data(src_fname, n_folds, splits, alphabet):
    word_folds = [{} for _ in range(n_folds)]
    sentence_folds = [[] for _ in range(n_folds)]
    with open(src_fname, 'r') as f:
        for i, line in tqdm(enumerate(f), desc='Processing wiki data',
                            total=len(splits)):
            if i in splits:
                fold = splits[i]
                process_line(line, word_folds[fold], sentence_folds[fold], alphabet)
    return word_folds, sentence_folds


def count_tokens(folds):
    return [sum([x['count'] for x in word_info.values()]) for word_info in folds]


def count_types(folds):
    return [len(word_info) for word_info in folds]


def process(src_fname, tgt_fname, n_folds, max_sentences=None):
    # spacy_tokenizer = load_spacy(spacy_option)
    n_sentences = count_sentences(src_fname)
    splits = get_fold_splits(n_sentences, n_folds, max_sentences=max_sentences)
    alphabet = Alphabet()

    word_folds, sentence_folds = process_data(src_fname, n_folds, splits, alphabet)
    n_tokens = count_tokens(word_folds)
    n_types = count_types(word_folds)
    util.write_data(tgt_fname, (word_folds, sentence_folds, alphabet, n_tokens))

    print('# unique chars:', len(alphabet))
    print('# tokens per fold:', n_tokens)
    print('# types per fold:', n_types)


def main():
    args = get_args()
    logging.info(args)

    process(args.wikipedia_tokenized_file, args.data_file, args.n_folds, args.max_sentences)


if __name__ == '__main__':
    main()
