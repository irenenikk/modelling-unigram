import os
import sys
import torch

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.train import load_generator
from h02_learn.adaptor import Adaptor
from util import constants, argparser

def get_args():
    argparser.add_argument('--epochs', type=int, default=5)
    # Save
    argparser.add_argument('--generator-path', type=str)
    argparser.add_argument('--adaptor-results-file', type=str, required=True)
    # adaptor
    argparser.add_argument('--adaptor-iterations', type=int, default=10)
    argparser.add_argument('--adaptor-state-file', type=str, required=True)
    argparser.add_argument('--load-adaptor-init-state', default=False, action='store_true')
    args = argparser.parse_args()
    return args

def evaluate_adaptor(dataloader, generator, adaptor):
    print('Evaluating adaptor with a dataset of size', len(dataloader.dataset))
    generator.eval()
    dataloader.dataset.eval()
    with torch.no_grad():
        cross_entropy = adaptor.calculate_cross_entropy(dataloader, generator)
    generator.train()
    dataloader.dataset.train()
    return cross_entropy

def save_pitman_yor_results(model, alpha, beta, train_loss, dev_loss, test_loss, test_size):
    results_fname = args.adaptor_results_file
    print('Saving to', results_fname)
    results = []
    file_size = os.path.getsize(results_fname)
    if file_size == 0:
        results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                    'dropout_p', 'alpha', 'beta', 'train_loss', 'dev_loss', 'test_loss', 'test_size']]
    results += [[model.alphabet_size, model.embedding_size, model.hidden_size, model.nlayers,\
                model.dropout_p, alpha, beta, train_loss, dev_loss, test_loss, test_size]]
    util.write_csv(results_fname, results)

def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds,\
                                        args.batch_size, test=True)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    generator = load_generator(alphabet, args.generator_path)
    generator.eval()
    adaptor = Adaptor(0, 0, alphabet, trainloader, state_filename=args.adaptor_state_file,\
                    load_state=True, save_state=False)

    train_loss = evaluate_adaptor(trainloader, generator, adaptor)
    dev_loss = evaluate_adaptor(devloader, generator, adaptor)
    test_loss = evaluate_adaptor(testloader, generator, adaptor)

    print('Adaptor Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    save_pitman_yor_results(generator, adaptor.alpha, adaptor.beta, train_loss, dev_loss, test_loss, len(testloader.dataset))


if __name__ == '__main__':
    main()
