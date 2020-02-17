import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from util import util
from .types import TypeDataset
from .tokens import TokenDataset


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
        sentence = item[0]
        sent_len = len(sentence) - 1  # Does not need to predict SOS
        x[i, :sent_len] = sentence[:-1]
        y[i, :sent_len] = sentence[1:]

    if len(batch[0]) == 3:
        # return the index in training
        return x, y, torch.Tensor([b[2] for b in batch])

    weights = torch.cat([entry[1] for entry in batch])
    return x, y, weights



def get_data_cls(data_type):
    if data_type == 'types':
        return TypeDataset
    if data_type == 'tokens':
        return TokenDataset
    raise ValueError('Invalid data requested %s' % data_type)


def load_data(fname):
    return util.read_data(fname)


def get_alphabet(data):
    _, alphabet, _ = data
    return alphabet

def get_data_loader(data_file, data_type, batch_size, subset_size=None):
    dataset_cls = get_data_cls(data_type)
    data = load_data(data_file)
    # the fold doesn't matter here
    dataset = dataset_cls(data, range(10))
    # this allows only using a subset of the dataset in development
    indice_amount = subset_size if subset_size is not None else len(dataset)
    subset_indices = np.random.choice(len(dataset), indice_amount, replace=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=generate_batch, 
                            sampler=SubsetRandomSampler(subset_indices))
    return dataloader


def get_data_loader_with_folds(dataset_cls, data, folds, batch_size, shuffle):
    trainset = dataset_cls(data, folds)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                             collate_fn=generate_batch)
    return trainloader


def get_data_loaders_with_folds(data_type, fname, folds, batch_size):
    dataset_cls = get_data_cls(data_type)
    data = load_data(fname)
    alphabet = get_alphabet(data)
    trainloader = get_data_loader_with_folds(dataset_cls, data, folds[0], batch_size=batch_size, shuffle=True)
    devloader = get_data_loader_with_folds(dataset_cls, data, folds[1], batch_size=batch_size, shuffle=False)
    testloader = get_data_loader_with_folds(dataset_cls, data, folds[2], batch_size=batch_size, shuffle=False)
    return trainloader, devloader, testloader, alphabet
