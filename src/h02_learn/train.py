import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import LstmLM
from h02_learn.train_info import TrainInfo
from util import argparser
from util import util
from util import constants



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

    args = argparser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    return args

# np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])


def get_model(alphabet_size, args):
    return LstmLM(
        alphabet_size, args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout)


def train_batch(x, y, model, optimizer, criterion):
    optimizer.zero_grad()
    x, y = x.to(device=constants.device), y.to(device=constants.device)
    y_hat = model(x)
    loss = criterion(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))
    loss.backward()
    optimizer.step()

    return loss.item()


def _evaluate(evalloader, model, alphabet):
    criterion = nn.CrossEntropyLoss(ignore_index=alphabet.char2idx('PAD'), reduction='none') \
        .to(device=constants.device)

    dev_loss, n_instances = 0, 0
    for x, y, weights in evalloader:
        x, y = x.to(device=constants.device), y.to(device=constants.device)
        y_hat = model(x)
        loss = criterion(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))\
            .reshape_as(y).sum(-1)
        dev_loss += (loss * weights).sum()
        n_instances += weights.sum()

    return dev_loss / n_instances


def evaluate(evalloader, model, alphabet):
    model.eval()
    evalloader.dataset.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model, alphabet)
    model.train()
    evalloader.dataset.train()
    return result


def train(trainloader, devloader, model, alphabet, eval_batches, wait_iterations):
    # optimizer = optim.AdamW(model.parameters())
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=alphabet.char2idx('PAD')) \
        .to(device=constants.device)
    train_info = TrainInfo(wait_iterations, eval_batches)

    while not train_info.finish:
        for x, y in trainloader:
            loss = train_batch(x, y, model, optimizer, criterion)
            train_info.new_batch(loss)

            if train_info.eval:
                dev_loss = evaluate(devloader, model, alphabet)

                if train_info.is_best(dev_loss):
                    model.set_best()
                elif train_info.finish:
                    break

                train_info.print_progress(dev_loss)

    model.recover_best()


def save_results(model, train_loss, dev_loss, test_loss, results_fname):
    results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                'dropout_p', 'train_loss', 'dev_loss', 'test_loss']]
    results += [[model.alphabet_size, model.embedding_size, model.hidden_size,
                 model.nlayers, model.dropout_p,
                 train_loss, dev_loss, test_loss]]
    util.write_csv(results_fname, results)


def save_checkpoints(model, train_loss, dev_loss, test_loss, checkpoints_path):
    model.save(checkpoints_path)
    results_fname = checkpoints_path + '/results.csv'
    save_results(model, train_loss, dev_loss, test_loss, results_fname)


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    model = get_model(len(alphabet), args).to(device=constants.device)
    train(trainloader, devloader, model, alphabet, args.eval_batches, args.wait_iterations)

    train_loss = evaluate(trainloader, model, alphabet)
    dev_loss = evaluate(devloader, model, alphabet)
    test_loss = evaluate(testloader, model, alphabet)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    save_checkpoints(model, train_loss, dev_loss, test_loss, args.checkpoints_path)


if __name__ == '__main__':
    main()
