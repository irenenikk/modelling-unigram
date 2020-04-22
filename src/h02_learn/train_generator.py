import sys
import torch.nn as nn
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.model import LstmLM
from h02_learn.train_info import TrainInfo
from h03_eval.eval_generator import evaluate_generator
from util.argparser import get_argparser, parse_args
from util import util
from util import constants


def get_args():
    # Optimization
    argparser = get_argparser()
    argparser.add_argument('--eval-batches', type=int, default=200)
    argparser.add_argument('--wait-epochs', type=int, default=5)
    # Save
    argparser.add_argument('--generator-path', type=str)
    argparser.add_argument('--max-train-tokens', type=int)
    args = parse_args(argparser)
    args.wait_iterations = args.wait_epochs * args.eval_batches
    return args

def load_generator(alphabet, checkpoints_path):
    generator = LstmLM.load(checkpoints_path)
    generator.ignore_index = alphabet.char2idx('PAD')
    generator.to(device=constants.device)
    return generator

def get_model(alphabet_size, args):
    return LstmLM(
        alphabet_size, args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout) \
        .to(device=constants.device)


def train_batch(x, y, model, optimizer, criterion):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1)).sum(-1)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(trainloader, devloader, model, alphabet, eval_batches, wait_iterations):
    # optimizer = optim.AdamW(model.parameters())
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=alphabet.char2idx('PAD'), reduction='none') \
        .to(device=constants.device)
    train_info = TrainInfo(wait_iterations, eval_batches)

    while not train_info.finish:
        for x, y, _, _ in trainloader:
            loss = train_batch(x, y, model, optimizer, criterion)
            train_info.new_batch(loss)

            if train_info.eval:
                dev_loss = evaluate_generator(devloader, model, alphabet)

                if train_info.is_best(dev_loss):
                    model.set_best()
                elif train_info.finish:
                    break

                train_info.print_progress(dev_loss)

    model.recover_best()
    return loss, dev_loss


def save_results(model, train_loss, dev_loss, train_size, dev_size, results_fname):
    results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',\
                'dropout_p', 'train_loss', 'dev_loss',\
                'train_size', 'dev_size']]
    results += [[model.alphabet_size, model.embedding_size, model.hidden_size,\
                 model.nlayers, model.dropout_p, train_loss, dev_loss,\
                 train_size, dev_size]]
    util.write_csv(results_fname, results)


def save_checkpoints(model, train_loss, dev_loss, train_size, dev_size, checkpoints_path):
    model.save(checkpoints_path)
    results_fname = checkpoints_path + '/results.csv'
    save_results(model, train_loss, dev_loss, train_size, dev_size, results_fname)

def main():
    # pylint: disable=all
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds,\
                                        args.batch_size, max_train_tokens=args.max_train_tokens)

    print('Train size: %d Dev size: %d ' %
          (len(trainloader.dataset), len(devloader.dataset)))

    model = get_model(len(alphabet), args)
    train(trainloader, devloader, model, alphabet, args.eval_batches, args.wait_iterations)

    train_loss = evaluate_generator(trainloader, model, alphabet)
    dev_loss = evaluate_generator(devloader, model, alphabet)

    print('Final Training loss: %.4f Dev loss: %.4f ' %
          (train_loss, dev_loss))

    save_checkpoints(model, train_loss, dev_loss, len(trainloader.dataset),\
                        len(devloader.dataset), args.generator_path)

if __name__ == '__main__':
    main()
