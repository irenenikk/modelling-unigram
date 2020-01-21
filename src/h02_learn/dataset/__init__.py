from torch.utils.data import DataLoader

from util import util
from .types import TypeDataset


def generate_batch(batch):
    r"""
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.[len(entry[0][0]) for entry in batch]
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        offsets: the offsets is a tensor of delimiters to represent the beginning
            index of the individual sequence in the text tensor.
        cls: a tensor saving the labels of individual text entries.
    """
    # text = torch.tensor([entry[0] for entry in batch])
    # label = torch.tensor([entry[0] for entry in batch])
    # text = [entry[0][0] for entry in batch]
    # text.zeros_like()

    tensor = batch[0]
    batch_size = len(batch)
    max_length = max([len(entry) for entry in batch]) - 1  # Does not need to predict SOS

    x = tensor.new_zeros(batch_size, max_length)
    y = tensor.new_zeros(batch_size, max_length)

    for i, sentence in enumerate(batch):
        sent_len = len(sentence) - 1  # Does not need to predict SOS
        x[i, :sent_len] = sentence[:-1]
        y[i, :sent_len] = sentence[1:]

    return x, y


def load_data(fname):
    return util.read_data(fname)


def get_alphabet(data):
    _, alphabet, _ = data
    return alphabet


def get_data_loader(fname, folds, batch_size, shuffle):
    trainset = TypeDataset(fname, folds)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                             collate_fn=generate_batch)
    return trainloader


def get_data_loaders(fname, folds, batch_size):
    data = load_data(fname)
    alphabet = get_alphabet(data)
    trainloader = get_data_loader(data, folds[0], batch_size=batch_size, shuffle=True)
    devloader = get_data_loader(data, folds[1], batch_size=batch_size, shuffle=False)
    testloader = get_data_loader(data, folds[2], batch_size=batch_size, shuffle=False)
    return trainloader, devloader, testloader, alphabet
