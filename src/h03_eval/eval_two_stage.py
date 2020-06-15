import os
import sys

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.train_generator import load_generator
from h02_learn.model.adaptor import Adaptor
from h02_learn.train_two_stage import evaluate_adaptor
from util import util
from util.argparser import get_argparser, parse_args, add_data_args

def get_args():
    argparser = get_argparser()
    # Save
    argparser.add_argument('--adaptor-results-file', type=str, required=True)
    # adaptor
    argparser.add_argument('--two-stage-state-folder', type=str, required=True)
    add_data_args(argparser)
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

def main():
    # pylint: disable=all
    args = get_args()
    folds = util.get_folds()

    trainloader, devloader, testloader, _ = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    generator = load_generator(args.two_stage_state_folder)
    generator.eval()
    adaptor = Adaptor.load(args.two_stage_state_folder)

    train_loss = evaluate_adaptor(trainloader, generator, adaptor)
    dev_loss = evaluate_adaptor(devloader, generator, adaptor)
    test_loss = evaluate_adaptor(testloader, generator, adaptor)

    print('Adaptor Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    alpha = adaptor.state['alpha']
    beta = adaptor.state['beta']
    save_pitman_yor_results(generator, args.dataset, alpha, beta, train_loss, dev_loss, test_loss,\
                            len(testloader.dataset), args.adaptor_results_file)


if __name__ == '__main__':
    main()
