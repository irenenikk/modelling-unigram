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

def get_model(alphabet, args):
    return LstmLM(
        len(alphabet), args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout, ignore_index=alphabet.char2idx('PAD')) \
        .to(device=constants.device)

def adaptor_loss(dataloader, generator, adaptor, alphabet):
    entropy = 0
    for x, y, weights in dataloader:
        word = ''.join(alphabet.idx2word(x))
        word_prob = adaptor.get_token_probability(generator, word)
        entropy += word_prob * np.log(word_prob)
    return entropy

def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    model = get_model(alphabet, args)
    token_data_loader = get_data_loader(args.data_file, 'tokens', 1, subset_size=10000)
    a = 1
    b = 1
    adaptor = Adaptor(a, b, alphabet, token_data_loader)
    # load generator
    model_path = os.path.join(args.checkpoints_path)
    LstmLM.load(model_path)
    # do we need to call this?
    #model.eval()
    # train adaptor
    adaptor.fit(model)

    generator_train_loss = evaluate(trainloader, model, alphabet)
    generator_dev_loss = evaluate(devloader, model, alphabet)
    generator_test_loss = evaluate(testloader, model, alphabet)

    adaptor_train_loss = adaptor_loss(trainloader, model, adaptor, alphabet)
    adaptor_dev_loss = adaptor_loss(devloader, model, adaptor, alphabet)
    adaptor_test_loss = adaptor_loss(testloader, model, adaptor, alphabet)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    save_checkpoints(model, train_loss, dev_loss, test_loss, args.checkpoints_path)


if __name__ == '__main__':
    main()
