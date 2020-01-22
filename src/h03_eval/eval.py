import sys

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import LstmLM
from h02_learn.train import evaluate
from util import argparser
from util import util
from util import constants


def get_args():
    # Data
    argparser.add_argument('--dataset', type=str)
    argparser.add_argument('--data-file', type=str)
    argparser.add_argument('--batch-size', type=int, default=512)
    # Models
    argparser.add_argument('--eval-path', type=str, required=True)
    # Save
    argparser.add_argument('--results-file', type=str, required=True)

    return argparser.parse_args()


def load_model(fpath):
    return LstmLM.load(fpath).to(device=constants.device)


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

    dataloaders = {
        dataset: get_data_loaders(dataset, args.data_file, folds, args.batch_size)
        for dataset in datasets
    }
    model_paths = util.get_dirs(args.eval_path)
    # model = load_model(args.model_path)
    # print(util.get_dirs(args.eval_path))
    # sys.exit()
    for dataset, dataloader in dataloaders.items():
        trainloader, devloader, testloader, _ = dataloader
        print('Dataset: %s Train size: %d Dev size: %d Test size: %d' %
              (dataset, len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    results = eval_all(model_paths, dataloaders)
    util.write_csv(args.results_file, results)
    # save_checkpoints(model, train_loss, dev_loss, test_loss, args.checkpoints_path)


if __name__ == '__main__':
    main()
