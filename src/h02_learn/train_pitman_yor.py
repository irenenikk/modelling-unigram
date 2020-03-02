import os
import sys
import torch

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.model import LstmLM
from h02_learn.train import get_args, evaluate
from util import util
from util import constants
from adaptor import Adaptor

def get_model(alphabet, args):
    return LstmLM(
        len(alphabet), args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout, ignore_index=alphabet.char2idx('PAD')) \
        .to(device=constants.device)

def evaluate_adaptor(dataloader, generator, adaptor):
    print('Evaluating adaptor with a dataset of size', len(dataloader.dataset))
    generator.eval()
    dataloader.dataset.eval()
    with torch.no_grad():
        cross_entropy = adaptor.calculate_cross_entropy(dataloader, generator)
    generator.train()
    dataloader.dataset.train()
    return cross_entropy

def save_pitman_yor_results(model, alpha, beta, train_loss, dev_loss, test_loss, results_fname):
    print('Saving to', results_fname)
    results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                'dropout_p', 'alpha', 'beta', 'train_loss', 'dev_loss', 'test_loss']]
    results += [[model.alphabet_size, model.embedding_size, model.hidden_size,
                 model.nlayers, model.dropout_p, alpha, beta,
                 train_loss, dev_loss, test_loss]]
    util.write_csv(results_fname, results)


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    alpha = 0.5
    beta = 0.5
    adaptor = Adaptor(alpha, beta, alphabet, trainloader)
    # load generator
    model_path = os.path.join(args.checkpoints_path)
    generator = LstmLM.load(model_path)
    generator.train()
    # train adaptor
    adaptor.load_fitted_adaptor()

    print('Getting generator training loss')
    generator_train_loss = evaluate(trainloader, generator, alphabet)
    print('Getting generator dev loss')
    generator_dev_loss = evaluate(devloader, generator, alphabet)
    print('Getting generator test loss')
    generator_test_loss = evaluate(testloader, generator, alphabet)

    print('Generator Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (generator_train_loss, generator_dev_loss, generator_test_loss))

    adaptor_train_loss = evaluate_adaptor(trainloader, generator, adaptor)
    adaptor_dev_loss = evaluate_adaptor(devloader, generator, adaptor)
    adaptor_test_loss = evaluate_adaptor(testloader, generator, adaptor)

    print('Adaptor Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (adaptor_train_loss, adaptor_dev_loss, adaptor_test_loss))

    save_pitman_yor_results(generator, alpha, beta, adaptor_train_loss, adaptor_dev_loss,\
                            adaptor_test_loss, args.checkpoints_path + '/results_pitman_yor.csv')


if __name__ == '__main__':
    main()
