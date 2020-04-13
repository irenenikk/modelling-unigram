import argparse
from . import util

parser = argparse.ArgumentParser(description='LanguageModel')

# Data defaults
parser.add_argument('--dataset', type=str)
parser.add_argument('--data-file', type=str)
parser.add_argument('--batch-size', type=int, default=32)

# Model defaults
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--embedding-size', type=int, default=128)
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--dropout', type=float, default=.33)

parser.add_argument('--seed', type=int, default=7,
                    help='Seed for random algorithms repeatability (default: 7)')

def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)


def get_default(*args, **kwargs):
    return parser.get_default(*args, **kwargs)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)

    # util.mkdir(args.rfolder)
    # util.mkdir(args.cfolder)
    util.config(args.seed)
    return args
