import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.model import LstmLM
from h02_learn.train_info import TrainInfo
# from h03_eval.eval_generator import evaluate_generator
from util.argparser import get_argparser, parse_args
from util import util
from util import constants


def get_args():
    argparser = get_argparser()
    # Optimization
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
    return generator


def get_model(alphabet, args):
    return LstmLM(
        len(alphabet), args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout,
        ignore_index=alphabet.PAD_IDX) \
        .to(device=constants.device)


def train_batch(x, y, model, optimizer):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = model.get_loss(y_hat, y).sum()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(trainloader, devloader, model, eval_batches, wait_iterations):
    optimizer = optim.AdamW(model.parameters())
    train_info = TrainInfo(wait_iterations, eval_batches)

    while not train_info.finish:
        for x, y, _, _ in trainloader:
            loss = train_batch(x, y, model, optimizer)
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
    for x, y, weights, _ in evalloader:
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
    # pylint: disable=all
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds,\
                                        args.batch_size, max_train_tokens=args.max_train_tokens)

    print('Train size: %d Dev size: %d ' %
          (len(trainloader.dataset), len(devloader.dataset)))

    model = get_model(alphabet, args)
    # criterion = nn.CrossEntropyLoss(ignore_index=alphabet.PAD_IDX, reduction='none') \
    #     .to(device=constants.device)

    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations)

    train_loss = evaluate(trainloader, model)
    dev_loss = evaluate(devloader, model)

    print('Final Training loss: %.4f Dev loss: %.4f ' %
          (train_loss, dev_loss))

    save_checkpoints(model, train_loss, dev_loss, len(trainloader.dataset),\
                        len(devloader.dataset), args.generator_path)

if __name__ == '__main__':
    main()
