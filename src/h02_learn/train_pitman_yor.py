import os
import sys
import torch

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.model import LstmLM
from h02_learn.train import evaluate, train
from util import constants, argparser
from util import util
from adaptor import Adaptor

def get_args():
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
    # training the generator
    argparser.add_argument('--train-generator', default=False, action='store_true')

    args = argparser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    return args

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

def load_generator(alphabet, args):
    generator = LstmLM.load(args.checkpoints_path)
    generator.ignore_index = alphabet.char2idx('PAD')
    generator.train()
    return generator

def train_with_pitman_yor(trainloader, devloader, alphabet, args, alpha, beta, total_iters, adaptor_iters, train_generator=False):
    generator = get_model(alphabet, args)
    if not train_generator:
        generator = load_generator(alphabet, args)
    adaptor = Adaptor(alpha, beta, alphabet, trainloader)
    for i in range(total_iters):
        print('Iteration', i)
        # train generator
        if train_generator:
            print('Training the generator')
            train(trainloader, devloader, generator, alphabet, args.eval_batches, args.wait_iterations)
        # train adaptor
        print('Training the adaptor')
        adaptor.fit(generator, iterations=adaptor_iters)
        if i % 2:
            generator_dev_loss = evaluate(devloader, generator, alphabet)
            print('Generator dev loss', generator_dev_loss)
            adaptor_dev_loss = evaluate_adaptor(devloader, generator, adaptor)
            print('Adaptor dev loss', adaptor_dev_loss)
    return generator, adaptor

def tune_alpha_and_beta(trainloader, devloader, alphabet, args, alphas, betas, total_iters, adaptor_iters):
    for alpha in alphas:
        for beta in betas:
            generator, adaptor = train_with_pitman_yor(trainloader, devloader, alphabet, args,\
                                                        alpha, beta, total_iters, adaptor_iters)


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    alpha = 0.5
    beta = 0.5
    generator, adaptor = train_with_pitman_yor(trainloader, devloader, alphabet, args,\
                                                alpha, beta, 10, 10)

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
                            adaptor_test_loss, args.adaptor_results_file)


if __name__ == '__main__':
    main()
