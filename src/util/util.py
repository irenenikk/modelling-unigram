import os
import pathlib
import io
import csv
import pickle
from collections import defaultdict
import numpy as np
import torch

def get_folds():
    # define training set, development set and test set respectively
    # folds range from 0 to 9
    return [list(range(8)), [8], [9]]

def config(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(filename, results):
    with io.open(filename, 'a', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)

def overwrite_csv(filename, results):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)

def write_data(filename, embeddings):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def write_torch_data(filename, embeddings):
    with open(filename, "wb") as f:
        torch.save(embeddings, f)


def read_data(filename):
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def read_torch_data(filename):
    with open(filename, "rb") as f:
        embeddings = torch.load(f)
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

def create_int_defaultdict():
    return defaultdict(int)

def permute_dict(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    permuted_values = np.random.permutation(values)
    return dict(zip(keys, permuted_values))

def define_plot_style(sns, plt):
    sns.set_palette("muted")
    sns.set_context("notebook", font_scale=1.6)
    plt.rc('font', family='serif', serif='Times New Roman')
