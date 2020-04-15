import sys
import torch.nn as nn
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.model import LstmLM
from h02_learn.train_info import TrainInfo
from h03_eval.eval_generator import evaluate_generator
from util import argparser
from util import util
from util import constants


def get_args():
    # Optimization
    argparser.add_argument('--eval-batches', type=int, default=200)
    argparser.add_argument('--wait-epochs', type=int, default=5)
    # Save
    argparser.add_argument('--generator-path', type=str)

    args = argparser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    return args


def get_model(alphabet_size, args):
    return LstmLM(
        alphabet_size, args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout) \
        .to(device=constants.device)


def train_batch(x, y, model, optimizer, criterion):
    optimizer.zero_grad()
    x, y = x.to(device=constants.device), y.to(device=constants.device)
    y_hat = model(x)
    loss = criterion(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))
    loss.backward()
    optimizer.step()

    return loss.item()


def train(trainloader, devloader, model, alphabet, eval_batches, wait_iterations):
    # optimizer = optim.AdamW(model.parameters())
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=alphabet.char2idx('PAD')) \
        .to(device=constants.device)
    train_info = TrainInfo(wait_iterations, eval_batches)

    while not train_info.finish:
        for x, y, _ in trainloader:
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
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    model = get_model(len(alphabet), args)
    train(trainloader, devloader, model, alphabet, args.eval_batches, args.wait_iterations)

    train_loss = evaluate_generator(trainloader, model, alphabet)
    dev_loss = evaluate_generator(devloader, model, alphabet)
    test_loss = evaluate_generator(testloader, model, alphabet)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    save_checkpoints(model, train_loss, dev_loss, test_loss, args.checkpoints_path)


if __name__ == '__main__':
    main()
