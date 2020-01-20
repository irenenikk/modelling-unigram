import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import LstmLM
from util import argparser
from util import util
from util import constants


def get_args():
    # Data
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

def _evaluate(evalloader, model, criterion):
    loss, n_instances = 0, 0
    for x, y in evalloader:
        batch_size = x.shape[0]
        y_hat = model(x)
        loss += criterion(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1)) * batch_size
        n_instances += batch_size

    return loss / n_instances


def evaluate(evalloader, model, criterion):
    model.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model, criterion)
    model.train()
    return result


def print_progress(batch_id, best_batch, wait_iterations, running_loss, dev_loss):
    avg_loss = sum(running_loss) / len(running_loss)
    max_epochs = best_batch + wait_iterations
    print('(%05d/%05d) Training loss: %.4f Dev loss: %.4f' %
          (batch_id / 100, max_epochs / 100, avg_loss, dev_loss))


def train(trainloader, devloader, model, criterion, eval_batches, wait_iterations):
    optimizer = optim.AdamW(model.parameters())
    batch_id, running_loss = 0, []
    best_loss, best_batch = float('inf'), 0

    while (batch_id - best_batch) < wait_iterations:
        for x, y in trainloader:
            loss = train_batch(x, y, model, optimizer, criterion)

            batch_id += 1
            running_loss += [loss]
            if (batch_id % eval_batches) == 0:
                dev_loss = evaluate(devloader, model, criterion)

                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_batch = batch_id
                    model.set_best()
                elif (batch_id - best_batch) >= wait_iterations:
                    break

                print_progress(batch_id, best_batch, wait_iterations, running_loss, dev_loss)
                running_loss = []

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
        get_data_loaders(args.data_file, folds, args.batch_size)

    model = get_model(len(alphabet), args).to(device=constants.device)
    criterion = nn.CrossEntropyLoss(ignore_index=alphabet.char2idx('PAD'))
    train(trainloader, devloader, model, criterion, args.eval_batches, args.wait_iterations)

    train_loss = evaluate(trainloader, model, criterion)
    dev_loss = evaluate(devloader, model, criterion)
    test_loss = evaluate(testloader, model, criterion)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    save_checkpoints(model, train_loss, dev_loss, test_loss, args.checkpoints_path)



if __name__ == '__main__':
    main()
