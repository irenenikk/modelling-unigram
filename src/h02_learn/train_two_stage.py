import time
import sys
import os
import copy
import torch
from tqdm import tqdm

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds, get_data_loader
from h02_learn.train_generator import train, load_generator
from h02_learn.train_generator import evaluate as evaluate_generator
from h02_learn.dataset.table_label import TableLabelDataset
from h02_learn.model.adaptor import Adaptor
from h02_learn.dataset.types_from_tokens import TypesFromTokensDataset
from util.argparser import get_argparser, parse_args, add_all_defaults
from util import util


def get_args():
    argparser = get_argparser()
    # Save
    argparser.add_argument('--adaptor-results-file', type=str, required=True)
    # adaptor
    argparser.add_argument('--alpha', type=float, required=True)
    argparser.add_argument('--beta', type=float, required=True)
    argparser.add_argument('--adaptor-iterations', type=int, default=6)
    argparser.add_argument('--two-stage-state-folder', type=str, required=True)
    argparser.add_argument('--load-adaptor-init-state', default=False, action='store_true')

    add_all_defaults(argparser)
    args = parse_args(argparser)
    args.wait_iterations = args.wait_epochs * args.eval_batches
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

def train_adaptor(adaptor, generator, types_logprobs, trainloader, devloader, adaptor_iterations):
    """ Train and recover the state performing the best on development set """
    min_dev_loss = float('inf')
    best_state = copy.deepcopy(adaptor.state)
    best_tables_with_word_labels = None
    for adaptor_iter in range(adaptor_iterations):
        print('Adaptor iteration', adaptor_iter + 1, '/', adaptor_iterations)
        tables_with_word_labels = adaptor.fit(types_logprobs, trainloader)
        two_stage_dev_loss = evaluate_adaptor(devloader, generator, adaptor)
        print('Adaptor dev loss', two_stage_dev_loss)
        if two_stage_dev_loss < min_dev_loss:
            min_dev_loss = two_stage_dev_loss
            best_state = copy.deepcopy(adaptor.state)
            best_tables_with_word_labels = copy.deepcopy(tables_with_word_labels)
    adaptor.set_state(best_state)
    return best_tables_with_word_labels, min_dev_loss

def train_generator(generator, tables_with_word_labels, token_devloader,\
                    args, alphabet, save_state=True):
    generator.train()
    tables_with_word_labels_dataset = TableLabelDataset(tables_with_word_labels, alphabet)
    table_label_dataloader = get_data_loader(tables_with_word_labels_dataset,\
                                                args.batch_size)
    _, generator_dev_loss = train(table_label_dataloader, token_devloader, generator,
                                  args.eval_batches, args.wait_iterations, args.dataset)
    if save_state:
        generator.save(args.two_stage_state_folder)
    print('Generator dev loss', generator_dev_loss)

def train_two_stage_model(generator, adaptor, token_trainloader, token_devloader, token_alphabet,
                          type_trainloader, args, save_state=True):
    tables_with_word_labels = adaptor.state['tables_with_word_label']
    for i in range(args.epochs):
        print('Iteration', i)
        # precalculate the type logprobs
        types_logprobs = precalculate_types_logprobs(generator, type_trainloader)
        # train adaptor
        print('Training the adaptor')
        tables_with_word_labels, two_stage_dev_loss = \
            train_adaptor(adaptor, generator, types_logprobs,
                          token_trainloader, token_devloader, args.adaptor_iterations)
        # train generator
        print('Training the generator with table label data')
        train_generator(generator, tables_with_word_labels,\
                        token_devloader, args, token_alphabet, save_state=save_state)
    return two_stage_dev_loss


def save_two_stage_training_results(model, args, train_loss, dev_loss, generator_dev_loss,\
                                        training_time, train_size, dev_size):
    results_fname = args.adaptor_results_file
    print('Saving to', results_fname)
    results = []
    file_size = os.path.getsize(results_fname) if os.path.exists(results_fname) else 0
    if file_size == 0:
        results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                    'dropout_p', 'alpha', 'beta', 'train_loss', 'dev_loss',\
                    'generator_dev_losss', 'total_epochs', 'adaptor_iterations',\
                    'training_time', 'train_size', 'dev_size']]
    results += [[model.alphabet_size, model.embedding_size, model.hidden_size, model.nlayers,\
                model.dropout_p, args.alpha, args.beta, train_loss, dev_loss,\
                generator_dev_loss, args.epochs, args.adaptor_iterations,\
                training_time, train_size, dev_size]]
    util.write_csv(results_fname, results)

def precalculate_types_logprobs(generator, type_dataloader):
    types_logprobs = {}
    generator.eval()
    with torch.no_grad():
        for x, y, _, _, tokens in tqdm(type_dataloader, total=len(type_dataloader), \
                            desc='Precalculating type logprobs', mininterval=.2):
            logprobs = generator.get_word_log_probability(x, y)
            for type_logprob, word in zip(logprobs, tokens):
                types_logprobs[word] = type_logprob
    return types_logprobs

def initiate_two_stage_training(token_trainloader, token_devloader, token_alphabet, args):
    types_from_tokens = TypesFromTokensDataset(token_trainloader)
    type_trainloader = get_data_loader(types_from_tokens, batch_size=64, shuffle=False)

    start = time.time()

    generator = load_generator(args.generator_path)
    adaptor = Adaptor(args.alpha, args.beta, token_alphabet, args.two_stage_state_folder)
    two_stage_dev_loss = \
        train_two_stage_model(generator, adaptor, token_trainloader, token_devloader,\
                                token_alphabet, type_trainloader, args)

    end = time.time()
    training_time = end - start
    return generator, adaptor, two_stage_dev_loss, training_time

def main():
    args = get_args()
    folds = util.get_folds()

    token_trainloader, token_devloader, _, token_alphabet = \
        get_data_loaders_with_folds('tokens', args.data_file, folds,\
                                        args.batch_size, max_train_tokens=args.max_train_tokens)

    print('Train size: %d Dev size: %d' %
          (len(token_trainloader.dataset), len(token_devloader.dataset)))

    generator, adaptor, two_stage_dev_loss, training_time = \
        initiate_two_stage_training(token_trainloader, token_devloader, token_alphabet, args)

    print('Getting generator training loss')
    generator_train_loss = evaluate_generator(token_trainloader, generator)
    print('Getting generator dev loss')
    generator_dev_loss = evaluate_generator(token_devloader, generator)

    print('Generator training loss: %.4f Dev loss: %.4f' %
          (generator_train_loss, generator_dev_loss))

    two_stage_train_loss = evaluate_adaptor(token_trainloader, generator, adaptor)

    print('Two-stage model training loss: %.4f Dev loss: %.4f' %
          (two_stage_train_loss, two_stage_dev_loss))

    save_two_stage_training_results(generator, args, two_stage_train_loss, two_stage_dev_loss,\
                                    generator_dev_loss, training_time,\
                                    len(token_trainloader.dataset), len(token_devloader.dataset))


if __name__ == '__main__':
    main()
