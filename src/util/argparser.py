import argparse
from . import util


def add_all_defaults(parser):
    add_data_args(parser)
    add_optimisation_args(parser)
    add_generator_args(parser)

def add_optimisation_args(parser):
    # Optimization
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval-batches', type=int, default=200)
    parser.add_argument('--wait-epochs', type=int, default=5)

def add_data_args(parser):
    # Data defaults
    parser.add_argument('--max-train-tokens', type=int)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dataset', type=str)

def add_generator_args(parser):
    # Model defaults
    parser.add_argument('--generator-path', type=str)
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=.33)

def get_argparser():
    parser = argparse.ArgumentParser(description='LanguageModel')
    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random algorithms repeatability (default: 7)')
    return parser

def parse_args(parser):
    args = parser.parse_args()
    # util.mkdir(args.rfolder)
    # util.mkdir(args.cfolder)
    util.config(args.seed)
    return args
