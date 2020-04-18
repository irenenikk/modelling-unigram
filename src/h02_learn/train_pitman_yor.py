import os
import sys
import torch
import time

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds, get_data_loader
from h02_learn.model import LstmLM
from h02_learn.train import train, save_checkpoints, load_generator
from h02_learn.dataset.table_label import TableLabelDataset
from h03_eval.eval_generator import evaluate_generator
from h03_eval.eval_adaptor import evaluate_adaptor
from util import constants, argparser
from util import util
from h02_learn.adaptor import Adaptor

def get_args():
    argparser.add_argument('--epochs', type=int, default=1)
    # Data
    argparser.add_argument('--train-num', type=int, default=None)
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
    argparser.add_argument('--alpha', type=float, required=True)
    argparser.add_argument('--beta', type=float, required=True)
    argparser.add_argument('--adaptor-iterations', type=int, default=6)
    argparser.add_argument('--adaptor-state-file', type=str, required=True)
    argparser.add_argument('--load-adaptor-init-state', default=False, action='store_true')
    args = argparser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    return args

def get_model(alphabet, args):
    return LstmLM(
        len(alphabet), args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout, ignore_index=alphabet.char2idx('PAD')) \
        .to(device=constants.device)

def train_with_pitman_yor(trainloader, devloader, alphabet, epochs, training_args):
    generator = load_generator(alphabet, training_args['generator_path'])
    generator.train()
    adaptor = Adaptor(training_args['alpha'], training_args['beta'], alphabet, trainloader, state_filename=training_args['adaptor_state_file'],\
                        load_state=training_args['load_adaptor_init_state'], save_state=training_args['save_adaptor_state'])
    tables_with_word_labels = adaptor.state['tables_with_word_label']
    for i in range(epochs):
        print('Iteration', i)
        # train generator
        if len(tables_with_word_labels) > 0:
            print('Training the generator with table label data')
            # us the dataset defined by the adaptor if present
            tables_with_word_labels_dataset = TableLabelDataset(tables_with_word_labels, alphabet)
            table_label_dataloader = get_data_loader(tables_with_word_labels_dataset, training_args['batch_size'])
            if training_args['reset_generator']:
                generator = get_model(alphabet, args)
            _, generator_dev_loss = train(table_label_dataloader, devloader, generator, alphabet, training_args['eval_batches'], training_args['wait_iterations'])
            generator.save(training_args['generator_path'] + '_retrained')
            print('Generator dev loss', generator_dev_loss)
        # train adaptor
        print('Training the adaptor')
        # TODO: refactor training to use early stopping based on dev loss?
        for adaptor_iter in range(training_args['adaptor_iterations']):
            print('Adaptor iteration', adaptor_iter + 1, '/', training_args['adaptor_iterations'])
            tables_with_word_labels = adaptor.fit(generator)
        adaptor_dev_loss = evaluate_adaptor(devloader, generator, adaptor)
        print('Adaptor dev loss', adaptor_dev_loss)
    return generator, adaptor, adaptor_dev_loss

def build_training_args(args, save_adaptor_state):
    training_args = vars(args)
    training_args['save_adaptor_state'] = save_adaptor_state
    return training_args

def save_pitman_yor_training_results(model, args, train_loss, dev_loss, generator_dev_loss, training_time, train_size, dev_size):
    results_fname = args.adaptor_results_file
    print('Saving to', results_fname)
    results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                'dropout_p', 'alpha', 'beta', 'train_loss', 'dev_loss',\
                'generator_dev_losss', 'total_epochs', 'adaptor_iterations', 'training_time', 'train_size', 'dev_size']]
    results += [[model.alphabet_size, model.embedding_size, model.hidden_size, model.nlayers,\
                model.dropout_p, args.alpha, args.beta, train_loss, dev_loss,\
                generator_dev_loss, args.epochs, args.adaptor_iterations, training_time, train_size, dev_size]]
    util.write_csv(results_fname, results)

def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size, args.train_num)

    trainset_size = len(trainloader.dataset)
    if args.train_num is not None and args.train_num < trainset_size:
        trainset_size = args.train_num

    print('Train size: %d Dev size: %d' %
          (len(trainloader.dataset), len(devloader.dataset)))

    start = time.time()

    training_args = build_training_args(args, save_adaptor_state=True)
    generator, adaptor, adaptor_dev_loss = train_with_pitman_yor(trainloader, devloader, alphabet, args.epochs, training_args)

    end = time.time()
    training_time = end - start

    print('Getting generator training loss')
    generator_train_loss = evaluate_generator(trainloader, generator, alphabet)
    print('Getting generator dev loss')
    generator_dev_loss = evaluate_generator(devloader, generator, alphabet)

    print('Generator Training loss: %.4f Dev loss: %.4f' %
          (generator_train_loss, generator_dev_loss))

    adaptor_train_loss = evaluate_adaptor(trainloader, generator, adaptor)

    print('Adaptor Training loss: %.4f Dev loss: %.4f' %
          (adaptor_train_loss, adaptor_dev_loss))

    save_pitman_yor_training_results(generator, args, adaptor_train_loss, adaptor_dev_loss,\
                                        generator_dev_loss, training_time, trainset_size, len(devloader.dataset))


if __name__ == '__main__':
    main()
