import os
import pathlib
import io
import csv
import pickle
import numpy as np
import torch


def config(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(filename, results):
    with io.open(filename, 'a', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def write_data(filename, embeddings):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def read_data(filename):
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def read_data_if_exists(filename):
    try:
        return read_data(filename)
    except FileNotFoundError:
        return {}


def remove_if_exists(fname):
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass


def get_filenames(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isfile(os.path.join(filepath, f))]
    return sorted(filenames)


def get_dirs(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isdir(os.path.join(filepath, f))]
    return sorted(filenames)


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

def hacked_exp(x):
    # the exp normalise trick to avoid over/underflowing:
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    x = np.asarray(x)
    maxim = x.max()
    y = np.exp(x - maxim)
    return y / y.sum()
