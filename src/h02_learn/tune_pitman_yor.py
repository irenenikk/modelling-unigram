import os
import sys
import torch

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.model import LstmLM
from h02_learn.train import evaluate, train
from h02_learn.train_pitman_yor import train_with_pitman_yor, evaluate_adaptor, load_generator, get_model
from util import constants, argparser
from util import util
from adaptor import Adaptor

def get_args():
    argparser.add_argument('--epochs', type=int, default=5)
    # Data
    argparser.add_argument('--dataset', type=str)
    argparser.add_argument('--data-file', type=str)
    argparser.add_argument('--batch-size', type=int, default=32)
    # Model
    argparser.add_argument('--nlayers', type=int, default=3)
    argparser.add_argument('--embedding-size', type=int, default=128)
    argparser.add_argument('--hidden-size', type=int, default=512)
    argparser.add_argument('--dropout', type=float, default=.33)
    # Optimization
    argparser.add_argument('--eval-batches', type=int, default=200)
    argparser.add_argument('--wait-epochs', type=int, default=5)
    # Save
    argparser.add_argument('--checkpoints-path', type=str)
    argparser.add_argument('--adaptor-results-file', type=str, required=True)
    # training options
    argparser.add_argument('--train-generator', default=False, action='store_true')
    # adaptor
    argparser.add_argument('--alphas', type=str, required=True)
    argparser.add_argument('--betas', type=str, required=True)
    argparser.add_argument('--adaptor-iterations', type=int, required=True)
    argparser.add_argument('--adaptor-state-file', type=str, required=True)
    args = argparser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    return args

def construct_pitman_yor_tuning_result_headers():
    results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                'dropout_p', 'alpha', 'beta', 'train_loss', 'dev_loss']]
    return results

def construct_pitman_yor_tuning_results(model, alpha, beta, train_loss, dev_loss):
    return [[model.alphabet_size, model.embedding_size, model.hidden_size,
                 model.nlayers, model.dropout_p, alpha, beta, train_loss, dev_loss]]

def tune_alpha_and_beta(trainloader, devloader, alphabet, args, alphas, betas, total_iters, adaptor_iters):
    best_loss = 1e5
    best_params = None
    tuning_results = construct_pitman_yor_tuning_result_headers()
    print('Tuning with alphas:', alphas, 'and betas', betas)
    for alpha in alphas:
        for beta in betas:
            generator, adaptor = train_with_pitman_yor(trainloader, devloader, alphabet, args,\
                                                        alpha, beta, total_iters, adaptor_iters)
            adaptor_train_loss = evaluate_adaptor(trainloader, generator, adaptor)
            adaptor_dev_loss = evaluate_adaptor(devloader, generator, adaptor)
            print('Adaptor train loss', adaptor_train_loss,'and dev loss', adaptor_dev_loss, 'with a =', alpha, ', b =', beta)
            if adaptor_dev_loss < best_loss:
                print('New best dev loss')
                best_loss = adaptor_dev_loss
                best_params = (alpha, beta)
            tuning_results += construct_pitman_yor_tuning_results(generator, alpha, beta, adaptor_train_loss, adaptor_dev_loss)
    print('Best dev loss', best_loss, 'obtained with (a, b) =', best_params)
    return tuning_results

def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    alphas = util.parse_string_to_list(args.alphas)
    betas = utils.parse_string_to_list(args.betas)
    tuning_results = tune_alpha_and_beta(trainloader, devloader, alphabet, args, alphas, betas, args.epochs, args.adaptor_iterations)
    print('Writing tuning results to', args.adaptor_results_file)
    util.write_csv(args.adaptor_results_file, tuning_results)

if __name__ == '__main__':
    main()
