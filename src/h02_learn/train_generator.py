import sys
import torch
# import torch.nn as nn
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.model import LstmLM
from h02_learn.train_info import TrainInfo
from util.argparser import get_argparser, parse_args, add_all_defaults
from util import util
from util import constants


def get_args():
    argparser = get_argparser()

    add_all_defaults(argparser)
    args = parse_args(argparser)
    args.wait_iterations = args.wait_epochs * args.eval_batches
    return args


def load_generator(checkpoints_path):
    return LstmLM.load(checkpoints_path)


def get_model(alphabet, args):
    return LstmLM(
        len(alphabet), args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout,
        ignore_index=alphabet.PAD_IDX) \
        .to(device=constants.device)


def train_batch(x, y, model, optimizer, by_character=False):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = model.get_loss(y_hat, y).sum(-1)
    if by_character:
        word_lengths = (y != 0).sum(-1)
        loss = (loss / word_lengths).mean()
    else:
        loss = loss.mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(trainloader, devloader, model, eval_batches, wait_iterations, dataset):
    optimizer = optim.AdamW(model.parameters())
    train_info = TrainInfo(wait_iterations, eval_batches)

    while not train_info.finish:
        for x, y, _, _, _ in trainloader:
            loss_per_char = dataset == 'sentences'
            loss = train_batch(x, y, model, optimizer, by_character=loss_per_char)
            train_info.new_batch(loss)

            if train_info.eval:
                dev_loss = evaluate(devloader, model)

                if train_info.is_best(dev_loss):
                    model.set_best()
                elif train_info.finish:
                    break

                train_info.print_progress(dev_loss)

    model.recover_best()
    return loss, dev_loss


def _evaluate(evalloader, model):
    dev_loss, n_instances = 0, 0
    for x, y, weights, _, _ in evalloader:
        y_hat = model(x)
        loss = model.get_loss(y_hat, y).sum(-1)
        dev_loss += (loss * weights).sum()
        n_instances += weights.sum()

    return (dev_loss / n_instances).item()


def evaluate(evalloader, model):
    model.eval()
    evalloader.dataset.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model)
    model.train()
    evalloader.dataset.train()
    return result


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
    args = get_args()
    folds = util.get_folds()

    trainloader, devloader, _, alphabet = get_data_loaders_with_folds(
        args.dataset, args.data_file, folds,
        args.batch_size, max_train_tokens=args.max_train_tokens)

    print('Train size: %d Dev size: %d ' %
          (len(trainloader.dataset), len(devloader.dataset)))

    model = get_model(alphabet, args)
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations, args.dataset)

    train_loss = evaluate(trainloader, model)
    dev_loss = evaluate(devloader, model)

    print('Final Training loss: %.4f Dev loss: %.4f ' %
          (train_loss, dev_loss))

    save_checkpoints(model, train_loss, dev_loss, len(trainloader.dataset),\
                        len(devloader.dataset), args.generator_path)

if __name__ == '__main__':
    main()
