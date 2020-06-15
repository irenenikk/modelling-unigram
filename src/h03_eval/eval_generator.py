import sys
import os

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.model import LstmLM
from h02_learn.train_generator import evaluate
from util.argparser import get_argparser, parse_args, add_data_args
from util import util
from util import constants


def get_args():
    argparser = get_argparser()
    # Models
    argparser.add_argument('--eval-path', type=str, required=True)
    # Save
    argparser.add_argument('--results-file', type=str, required=True)
    add_data_args(argparser)
    return parse_args(argparser)


def load_model(fpath):
    return LstmLM.load(fpath).to(device=constants.device)


def eval_all(model_paths, dataloaders):
    results = [['model', 'dataset', 'train_loss', 'dev_loss', 'test_loss']]
    for model_path in model_paths:
        if not os.path.exists(LstmLM.get_name(model_path)):
            continue
        model = load_model(model_path)
        model_name = model_path.split('/')[-1]

        for dataset, dataloader in dataloaders.items():
            trainloader, devloader, testloader, _ = dataloader
            print('Evaluating model: %s on dataset: %s' %
                  (model_name, dataset))

            train_loss = evaluate(trainloader, model)
            dev_loss = evaluate(devloader, model)
            test_loss = evaluate(testloader, model)

            print('Final %s Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
                  (dataset, train_loss, dev_loss, test_loss))

            results += [[model_name, dataset, train_loss, dev_loss, test_loss]]

    return results


def main():
    args = get_args()
    folds = util.get_folds()
    datasets = ['types', 'tokens']

    model_paths = util.get_dirs(args.eval_path)

    dataloaders = {
        dataset: get_data_loaders_with_folds(dataset, args.data_file, folds,\
                                             args.batch_size)
        for dataset in datasets
    }
    for dataset, dataloader in dataloaders.items():
        trainloader, devloader, testloader, _ = dataloader
        print('Dataset: %s Train size: %d Dev size: %d Test size: %d' %
              (dataset, len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    results = eval_all(model_paths, dataloaders)
    util.overwrite_csv(args.results_file, results)


if __name__ == '__main__':
    main()
