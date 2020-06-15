import sys
import numpy as np

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds, get_data_loader
from h02_learn.train_two_stage import train_two_stage_model, load_generator, Adaptor
from h02_learn.dataset.types_from_tokens import TypesFromTokensDataset
from h03_eval.eval_two_stage import evaluate_adaptor
from util.argparser import get_argparser, parse_args, add_all_defaults
from util import util

def get_args():
    argparser = get_argparser()
    # Save
    argparser.add_argument('--results-file', type=str, required=True)
    # adaptor
    argparser.add_argument('--no-iterations', type=int, default=10)
    argparser.add_argument('--beta-limit', type=int)
    argparser.add_argument('--adaptor-iterations', type=int, default=6)
    argparser.add_argument('--two-stage-state-folder', type=str, required=True)
    add_all_defaults(argparser)
    args = parse_args(argparser)
    args.wait_iterations = args.wait_epochs * args.eval_batches
    return args

def construct_pitman_yor_tuning_result_headers():
    results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                'dropout_p', 'alpha', 'beta', 'train_loss', 'dev_loss']]
    return results

def construct_pitman_yor_tuning_results(model, alpha, beta, train_loss, dev_loss):
    return [[model.alphabet_size, model.embedding_size, model.hidden_size,\
                 model.nlayers, model.dropout_p, alpha, beta, train_loss, dev_loss]]

def tune_alpha_and_beta(trainloader, devloader, alphabet, args, iterations, beta_limit):
    # pylint: disable=too-many-locals
    best_loss = float('inf')
    types_from_tokens = TypesFromTokensDataset(trainloader)
    type_trainloader = get_data_loader(types_from_tokens, batch_size=64, shuffle=False)
    best_params = None
    tuning_results = construct_pitman_yor_tuning_result_headers()
    for _ in range(iterations):
        alpha = np.random.uniform(0, 1)
        beta = np.random.uniform(100, beta_limit)
        generator = load_generator(args.generator_path)
        adaptor = Adaptor(alpha, beta, alphabet, '', save_state=False)
        print('--------------------------------')
        print('alpha =', alpha, 'beta = ', beta)
        dev_loss = \
            train_two_stage_model(generator, adaptor, trainloader, devloader, \
                                    alphabet, type_trainloader, args, save_state=False)
        print('Adaptor dev loss', dev_loss, 'with a =', alpha, ', b =', beta)
        if dev_loss < best_loss:
            print('New best loss')
            best_loss = dev_loss
            best_params = (alpha, beta)
            print('Saving adaptor and generator state to', args.two_stage_state_folder)
            adaptor.save_fitted_state(args.two_stage_state_folder)
            generator.save(args.two_stage_state_folder)
        train_loss = evaluate_adaptor(trainloader, generator, adaptor)
        tuning_results += \
            construct_pitman_yor_tuning_results(generator, alpha, beta, train_loss, dev_loss)
    print('Best loss', best_loss, 'obtained with (a, b) =', best_params)
    return tuning_results

def main():
    # pylint: disable=all
    args = get_args()
    folds = util.get_folds()

    trainloader, devloader, _, alphabet = \
        get_data_loaders_with_folds('tokens', args.data_file, folds, args.batch_size,\
                                    max_train_tokens=args.max_train_tokens)
    print('Train size: %d Dev size %d' %
          (len(trainloader.dataset), len(devloader.dataset)))

    beta_limit = len(trainloader.dataset) * 2
    if args.beta_limit is not None:
        beta_limit = args.beta_limit

    print('Tuning alpha and beta')
    tuning_results = tune_alpha_and_beta(trainloader, devloader, alphabet, args, args.no_iterations, beta_limit)
    print('Writing tuning results to', args.results_file)
    util.write_csv(args.results_file, tuning_results)

if __name__ == '__main__':
    main()
