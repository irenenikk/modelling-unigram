import argparse
from . import util

parser = argparse.ArgumentParser(description='LanguageModel')
# # Data
# parser.add_argument('--data', type=str, default='celex',
#                     help='Dataset used. (default: celex)')
# parser.add_argument('--data-path', type=str, default='datasets',
#                     help='Path where data is stored.')

# Model
parser.add_argument('--model', default='lstm', choices=['lstm'],
                    help='Model used. (default: lstm)')
# parser.add_argument('--opt', action='store_true', default=False,
#                     help='Should use optimum parameters in training.')
# parser.add_argument('--train-mode', type=str, default='type',
#                     help='Training mode used. (default: type)')

# Others
# parser.add_argument('--results-path', type=str, default='results',
#                     help='Path where results should be stored.')
# parser.add_argument('--checkpoint-path', type=str, default='checkpoints',
#                     help='Path where checkpoints should be stored.')
# parser.add_argument('--csv-folder', type=str, default=None,
#                     help='Specific path where to save results.')
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
