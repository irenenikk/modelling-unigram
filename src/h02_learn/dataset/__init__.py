import torch
from torch.utils.data import DataLoader

from util import constants
from util import util
from .types import TypeDataset
from .tokens import TokenDataset
from .sentences import SentenceDataset


def generate_batch(batch):
    r"""
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.
    """
    tensor = batch[0][0]
    batch_size = len(batch)
    max_length = max([len(entry[0]) for entry in batch]) - 1  # Does not need to predict SOS

    x = tensor.new_zeros(batch_size, max_length)
    y = tensor.new_zeros(batch_size, max_length)

    for i, item in enumerate(batch):
        word = item[0]
        word_len = len(word) - 1  # Does not need to predict SOS
        x[i, :word_len] = word[:-1]
        y[i, :word_len] = word[1:]

    x, y = x.to(device=constants.device), y.to(device=constants.device)
    weights = torch.cat([b[1] for b in batch]).to(device=constants.device)
    indices = torch.LongTensor([b[2] for b in batch]).to(device=constants.device)
    tokens = [b[3] for b in batch]
    return x, y, weights, indices, tokens


def get_data_cls(data_type):
    if data_type == 'types':
        return TypeDataset
    if data_type == 'tokens':
        return TokenDataset
    if data_type == 'sentences':
        return SentenceDataset
    raise ValueError('Invalid data requested %s' % data_type)


def load_data(fname):
    return util.read_data(fname)


def get_alphabet(data):
    _, _, alphabet, _ = data
    return alphabet


def get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=generate_batch)
    return dataloader


def get_data_loader_with_folds(dataset_cls, data, folds, batch_size, shuffle, max_tokens=None):
    trainset = dataset_cls(data, folds, max_tokens)
    return DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, collate_fn=generate_batch)


def get_data_loaders_with_folds(data_type, fname, folds, batch_size, max_train_tokens=None):
    dataset_cls = get_data_cls(data_type)
    data = load_data(fname)
    alphabet = get_alphabet(data)
    trainloader = get_data_loader_with_folds(dataset_cls, data, folds[0],
                                             batch_size=batch_size, shuffle=True,
                                             max_tokens=max_train_tokens)
    devloader = get_data_loader_with_folds(dataset_cls, data, folds[1],
                                           batch_size=batch_size, shuffle=False)
    testloader = get_data_loader_with_folds(dataset_cls, data, folds[2],
                                            batch_size=batch_size, shuffle=False)
    return trainloader, devloader, testloader, alphabet
