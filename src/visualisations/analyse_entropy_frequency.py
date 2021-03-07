import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.append('./src/')
from util.argparser import get_argparser, parse_args
from util import util

def get_args():
    argparser = get_argparser()
    argparser.add_argument('--data-file', type=str, required=True)
    args = parse_args(argparser)
    return args

def get_cumulative_average(data, model):
    return data[model].expanding().mean()

def get_window_average(data, model):
    return data[model].rolling(1000).mean().dropna()

def get_average_per_freq_rolling(data, model):
    return data[[model, 'freq']].groupby('freq').mean().rolling(6).mean() # remember to take the window size into account when plotting

def get_average_per_freq(data, model):
    return data[[model, 'freq']].groupby('freq').mean()

def plot_scatter(x_values, sorted_data, column, color, jitter_lim=0.1, size=0.1):
    jittered = x_values + np.random.uniform(-jitter_lim, jitter_lim, len(x_values))
    plt.scatter(jittered, sorted_data[column], s=size, c=color)

def plot_data(x_axis, sorted_data, transform, x_label, scatter=True):
    type_color = 'b'
    token_color = 'g'
    two_stage_color = 'r'
    generator_color = '#D68910'
    fig = plt.figure(figsize=(8,4))
    if scatter:
        offset = 3 # this should be half the window size in case you use rolling average
        x_values = sorted_data['freq']+offset
        plot_scatter(x_values, sorted_data, 'type', type_color)
        plot_scatter(x_values, sorted_data, 'token', token_color)
        plot_scatter(x_values, sorted_data, 'generator', generator_color)
        plot_scatter(x_values, sorted_data, 'two_stage', two_stage_color)
        plt.ylim(4, 20)
        plt.xlim(offset, 1e4)
    plt.plot(x_axis, transform(sorted_data, 'type'),
             label='Type model', c=type_color)
    plt.plot(x_axis, transform(sorted_data, 'token'),
             label='Token model', c=token_color)
    plt.plot(x_axis, transform(sorted_data, 'generator'),
             label='Generator', c=generator_color)
    plt.plot(x_axis, transform(sorted_data, 'two_stage'),
             label='Two-stage model', c=two_stage_color)
    plt.xscale('log')
    plt.tight_layout()
    plt.xlabel(x_label)
    plt.ylabel('Word surprisal (nats)')
    #plt.legend()
    plt.show()
    fig.savefig('plot.png', bbox_inches='tight')

def compare_type_and_generator(data, n_vals=10):
    print('Comparing the generator and the type model')
    data['type_minus_gen'] = data['type'] - data['generator']
    gen_best = data.sort_values(by=['type_minus_gen'], ascending=False)
    gen_better = gen_best[gen_best['type_minus_gen'] > 0]
    gen_better_singletons = gen_better[gen_better['freq'] == 1]
    print('Generator is better')
    print(gen_better[:n_vals][['word', 'freq', 'type', 'generator']])
    print('With singletons')
    print(gen_better_singletons[:n_vals][['word', 'freq', 'type', 'generator']])

    type_better = gen_best[gen_best['type_minus_gen'] < 0]
    print('Type model is better')
    print(type_better[-n_vals:][['word', 'freq', 'type', 'generator']])
    print('With singletons')
    type_better_singletons = type_better[type_better['freq'] == 1]
    print(type_better_singletons[-n_vals:][['word', 'freq', 'type', 'generator']])


def top_k_average_entropies(freq_sorted, models, top_k=10000):
    print('Top', top_k, 'average')
    top_freq = freq_sorted[-top_k:]
    for model in models:
        avg = top_freq[model].mean()
        print(model, 'average', avg)

def singleton_average_entropies(data, models):
    print('Singletons')
    singletons = data[data['freq'] == 1]
    print('Singleton percentage', len(singletons)/len(data))
    for model in models:
        avg = singletons[model].mean()
        print(model, 'average', avg)

def non_singleton_average_entropies(data, models):
    print('Non-singletons')
    non_singletons = data[data['freq'] > 1]
    print('Non singleton percentage', len(non_singletons)/len(data))
    for model in models:
        avg = non_singletons[model].mean()
        print(model, 'average', avg)

def main():
    args = get_args()
    util.define_plot_style(sns, plt)

    data = pd.read_csv(args.data_file)
    data['rank'] += 1
    freq_sorted = data.sort_values(by=['freq'])

    #plot_data(rank_sorted['rank']/2, rank_sorted, get_cumulative_average,\
    #          'Word rank', scatter=False)
    plot_data(freq_sorted['freq'].unique(), freq_sorted, get_average_per_freq_rolling,\
              'Word frequency')

    print('test types', len(data))
    print('generator mean', data['generator'].mean())
    print('token mean', data['token'].mean())
    print('type mean', data['type'].mean())
    print('two-stage mean', data['two_stage'].mean())
    print('---------------------------------')

    compare_type_and_generator(data, 20)

    models = ['type', 'token', 'two_stage', 'generator']
    top_k_average_entropies(freq_sorted, models, top_k=10000)
    print('---------------------------------')
    singleton_average_entropies(data, models)
    print('---------------------------------')
    non_singleton_average_entropies(data, models)

if __name__ == '__main__':
    main()
