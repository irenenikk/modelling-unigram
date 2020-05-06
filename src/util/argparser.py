import argparse
from . import util

def add_data_args(parser):
    # Data defaults
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data-file', type=str)
    parser.add_argument('--batch-size', type=int, default=32)

def add_generator_args(parser):
    # Model defaults
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=.33)

def get_argparser():
    parser = argparse.ArgumentParser(description='LanguageModel')
    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random algorithms repeatability (default: 7)')
    add_data_args(parser)
    add_generator_args(parser)
    return parser

def parse_args(parser):
    args = parser.parse_args()
    # util.mkdir(args.rfolder)
    # util.mkdir(args.cfolder)
    util.config(args.seed)
    return args
