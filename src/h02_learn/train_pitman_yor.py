import sys
import torch
import os
import torch.nn as nn
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds, load_data
from h02_learn.model import LstmLM
from h02_learn.train_info import TrainInfo
from h02_learn.train import get_args, train_batch, train, save_checkpoints, save_results, evaluate
from util import argparser
from util import util
from util import constants
from adaptor import Adaptor

def get_model(alphabet, args):
    return LstmLM(
        len(alphabet), args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout, ignore_index=alphabet.char2idx('PAD')) \
        .to(device=constants.device)


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


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    model = get_model(alphabet, args)
    token_data, _, _ = load_data(args.data_file)
    adaptor = Adaptor(0.0001, 0.0001, alphabet, token_data)
    # load generator
    model_path = os.path.join(args.checkpoints_path)
    LstmLM.load(model_path)
    # do we need to call this?
    #model.eval()
    # train adaptor
    adaptor.fit(model)

    #train_loss = evaluate(trainloader, model, alphabet)
    #dev_loss = evaluate(devloader, model, alphabet)
    #test_loss = evaluate(testloader, model, alphabet)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    save_checkpoints(model, train_loss, dev_loss, test_loss, args.checkpoints_path)


if __name__ == '__main__':
    main()
