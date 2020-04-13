import os
import sys
import torch
import time
import numpy as np

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds, get_data_loader
from h02_learn.model import LstmLM
from h02_learn.train import train, save_checkpoints
from h02_learn.train_pitman_yor import train_with_pitman_yor, build_training_args
from h02_learn.dataset.table_label import TableLabelDataset
from h03_eval.eval_generator import evaluate_generator
from h03_eval.eval_adaptor import evaluate_adaptor
from util import constants, argparser
from util import util
from adaptor import Adaptor

def get_args():
    argparser.add_argument('--epochs', type=int, default=5)
    # Optimization
    argparser.add_argument('--eval-batches', type=int, default=200)
    argparser.add_argument('--wait-epochs', type=int, default=5)
    # Save
    argparser.add_argument('--generator-path', type=str)
    argparser.add_argument('--adaptor-results-file', type=str, required=True)
    # training options
    argparser.add_argument('--train-generator', default=False, action='store_true')
    argparser.add_argument('--reset-generator', default=False, action='store_true')
    # adaptor
    argparser.add_argument('--no-alphas', type=float, required=True)
    argparser.add_argument('--no-betas', type=int, required=True)
    argparser.add_argument('--beta-end', type=int)
    argparser.add_argument('--adaptor-iterations', type=int, default=10)
    argparser.add_argument('--adaptor-state-file', type=str, required=True)
    argparser.add_argument('--load-adaptor-init-state', default=False, action='store_true')
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

def tune_alpha_and_beta(trainloader, devloader, alphabet, args, alphas, betas):
    best_loss = 1e5
    best_params = None
    for alpha in alphas:
        for beta in betas:
            training_args = build_training_args(args, save_adaptor_state=False)
            training_args['alpha'] = alpha
            training_args['beta'] = beta
            generator, adaptor, adaptor_dev_loss = train_with_pitman_yor(trainloader, devloader, alphabet, args.epochs, training_args)
            print('Adaptor dev loss', adaptor_dev_loss, 'with a =', alpha, ', b =', beta)
            if adaptor_dev_loss < best_loss:
                print('New best loss')
                best_loss = adaptor_dev_loss
                best_params = (alpha, beta)
                print('Saving adaptor state to', args.adaptor_state_file)
                adaptor.save_fitted_adaptor(args.adaptor_state_file)
            tuning_results += construct_pitman_yor_tuning_results(generator, alpha, beta, adaptor_train_loss, adaptor_dev_loss)
    print('Best loss', best_loss, 'obtained with (a, b) =', best_params)
    return tuning_results

def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    beta_end = len(trainloader.dataset) * 10
    if args.beta_end is not None:
        beta_end = args.beta_end
    alphas = [x for x in np.arange(0, 1, float(1)/args.no_alphas)]
    betas = [x for x in np.arange(1, beta_end, beta_end/args.no_betas)]

    print('Tuning alpha and beta with values')
    print('alphas:', alphas)
    print('betas:', betas)
    tuning_results = tune_alpha_and_beta(trainloader, devloader, alphabet, args, alphas, betas)
    print('Writing tuning results to', args.adaptor_results_file)
    util.write_csv(args.adaptor_results_file, tuning_results)

if __name__ == '__main__':
    main()
