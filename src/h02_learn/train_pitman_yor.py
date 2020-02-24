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
from h02_learn.dataset import get_data_loader
from util import argparser
from util import util
from util import constants
from adaptor import Adaptor
import numpy as np

def get_model(alphabet, args):
    return LstmLM(
        len(alphabet), args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout, ignore_index=alphabet.char2idx('PAD')) \
        .to(device=constants.device)

def evaluate_adaptor(dataloader, generator, adaptor):
    generator.eval()
    dataloader.dataset.eval()
    with torch.no_grad():
        cross_entropy = adaptor.calculate_cross_entropy(dataloader, generator)
    generator.train()
    dataloader.dataset.train()
    return cross_entropy

def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    model = get_model(alphabet, args)
    a = 0.5
    b = 0.5
    # TODO: train inside a loop
    adaptor = Adaptor(a, b, alphabet, trainloader)
    # load generator
    # TODO: train the generator
    model_path = os.path.join(args.checkpoints_path)
    LstmLM.load(model_path)
    # TODO: do we need to call this?
    model.train()
    # train adaptor
    adaptor.fit(model)

    print('Getting generator training loss')
    generator_train_loss = evaluate(trainloader, model, alphabet)
    print('Getting generator dev loss')
    generator_dev_loss = evaluate(devloader, model, alphabet)
    print('Getting generator test loss')
    generator_test_loss = evaluate(testloader, model, alphabet)

    print('Generator Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (generator_train_loss, generator_dev_loss, generator_test_loss))
    adaptor_train_loss = evaluate_adaptor(trainloader, model, adaptor)
    adaptor_dev_loss = evaluate_adaptor(devloader, model, adaptor)
    adaptor_test_loss = evaluate_adaptor(testloader, model, adaptor)

    print('Adaptor Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (adaptor_train_loss, adaptor_dev_loss, adaptor_test_loss))

    save_checkpoints(model, train_loss, dev_loss, test_loss, args.checkpoints_path)


if __name__ == '__main__':
    main()
