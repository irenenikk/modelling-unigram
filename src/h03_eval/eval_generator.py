import sys
import torch
import torch.nn as nn

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.model import LstmLM
from util import argparser
from util import util
from util import constants


def get_args():
    # Data
    argparser.add_argument('--dataset', type=str)
    argparser.add_argument('--data-file', type=str)
    # Models
    argparser.add_argument('--eval-path', type=str, required=True)
    # Save
    argparser.add_argument('--results-file', type=str, required=True)

    return argparser.parse_args()


def load_model(fpath):
    return LstmLM.load(fpath).to(device=constants.device)

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

    return (dev_loss / n_instances).item()

def evaluate_generator(evalloader, model, alphabet):
    model.eval()
    evalloader.dataset.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model, alphabet)
    model.train()
    evalloader.dataset.train()
    return result

def eval_all(model_paths, dataloaders):
    results = [['model', 'dataset', 'train_loss', 'dev_loss', 'test_loss']]
    for model_path in model_paths:
        model = load_model(model_path)
        model_name = model_path.split('/')[-1]

        for dataset, dataloader in dataloaders.items():
            trainloader, devloader, testloader, alphabet = dataloader
            print('Evaluating model: %s on dataset: %s' %
                  (model_name, dataset))

            train_loss = evaluate(trainloader, model, alphabet)
            dev_loss = evaluate(devloader, model, alphabet)
            test_loss = evaluate(testloader, model, alphabet)

            print('Final %s Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
                  (dataset, train_loss, dev_loss, test_loss))

            results += [[model_name, dataset, train_loss, dev_loss, test_loss]]

    return results


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]
    datasets = ['types', 'tokens']

    model_paths = util.get_dirs(args.eval_path)

    dataloaders = {
        dataset: get_data_loaders_with_folds(dataset, args.data_file, folds, args.batch_size)
        for dataset in datasets
    }
    for dataset, dataloader in dataloaders.items():
        trainloader, devloader, testloader, _ = dataloader
        print('Dataset: %s Train size: %d Dev size: %d Test size: %d' %
              (dataset, len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    results = eval_all(model_paths, dataloaders)
    util.write_csv(args.results_file, results)


if __name__ == '__main__':
    main()
