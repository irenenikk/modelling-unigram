import os
import sys
from tqdm import tqdm

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.train_generator import load_generator, evaluate
from h02_learn.model.adaptor import Adaptor
from h03_eval.eval_generator import load_model
from util.argparser import get_argparser, parse_args
from util import util
from visualisations.zipfs_law import calculate_word_freqs, get_word_ranks

def get_args():
    argparser = get_argparser()
    argparser.add_argument('--max-train-tokens', type=int, required=True)
    argparser.add_argument('--data-language-dir', type=str, required=True)
    argparser.add_argument('--checkpoint-language-dir', type=str, required=True)
    argparser.add_argument('--alpha', type=str, required=True)
    argparser.add_argument('--beta', type=str, required=True)
    argparser.add_argument('--results-folder', type=str, required=True)
    args = parse_args(argparser)
    return args


def get_model(model_name, args):
    model_path = os.path.join(args.checkpoint_language_dir,
                              model_name + '_' + str(args.max_train_tokens))
    model = load_model(model_path)
    model.eval()
    return model


def get_lm_loss(model, x, y):
    y_hat = model(x)
    loss = model.get_loss(y_hat, y).sum(-1)
    return loss.item()


def get_two_stage_loss(adaptor, generator, x, y, word):
    generator_logprob = generator.get_word_log_probability(x, y)
    two_stage_logprob = adaptor.get_token_logprobability(generator_logprob, word)
    return -two_stage_logprob.item()

def two_stage_model_name(args):
    return 'two_stage_init_type_' + args.alpha.replace('.', '_') + \
           '_' + args.beta + '_' + str(args.max_train_tokens)

def get_word_freqs_and_ranks(data_file, folds):
    _, _, token_testloader, _ = get_data_loaders_with_folds(
        'tokens', data_file, folds, batch_size=1)
    word_freqs = calculate_word_freqs(token_testloader.dataset)
    word_ranks = get_word_ranks(token_testloader.dataset)
    return word_freqs, word_ranks

def calculate_surprisals(type_testloader, type_model, token_model,\
                         generator, adaptor, word_freqs, word_ranks):
    # pylint: disable=too-many-locals
    results = [['word', 'type', 'token', 'two_stage', 'generator', 'freq', 'rank']]
    for x, y, _, _, word_batch in tqdm(type_testloader, desc='Calculating type entropies',
                                       total=len(type_testloader.dataset)):
        word = word_batch[0]
        type_loss = get_lm_loss(type_model, x, y)
        token_loss = get_lm_loss(token_model, x, y)
        generator_loss = get_lm_loss(generator, x, y)
        two_stage_loss = get_two_stage_loss(adaptor, generator, x, y, word)
        freq = word_freqs[word]
        rank = word_ranks[word]
        results += [[word, type_loss, token_loss, two_stage_loss, generator_loss, freq, rank]]
    return results

def type_token_sanity_check(type_testloader, type_model, token_model):
    overall_type = evaluate(type_testloader, type_model)
    overall_token = evaluate(type_testloader, token_model)

    print('Overall type loss', overall_type)
    print('Overall token loss', overall_token)

def main():
    args = get_args()
    folds = util.get_folds()

    data_file = os.path.join(args.data_language_dir, 'processed.pckl')

    _, _, type_testloader, _ = get_data_loaders_with_folds(
        'types', data_file, folds, batch_size=1)

    word_freqs, word_ranks = get_word_freqs_and_ranks(data_file, folds)

    two_stage_state_folder = os.path.join(args.checkpoint_language_dir, two_stage_model_name(args))
    generator = load_generator(two_stage_state_folder)
    generator.eval()
    adaptor = Adaptor.load(two_stage_state_folder)

    token_model = get_model('tokens', args)
    type_model = get_model('types', args)

    results = calculate_surprisals(type_testloader, type_model, token_model,\
                                   generator, adaptor, word_freqs, word_ranks)

    results_file = os.path.join(args.results_folder, 'entropy_freq.csv')
    util.overwrite_csv(results_file, results)

if __name__ == '__main__':
    main()
