import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.append('./src/')
from util.argparser import get_argparser, parse_args

def get_args():
    argparser = get_argparser()
    argparser.add_argument('--data-file', type=str, required=True)
    args = parse_args(argparser)
    return args

def get_cumulative_average(data, model):
    return data[model].expanding().mean()

def get_window_average(data, model):
    return data[model].rolling(1000).mean().dropna()

def get_average_per_freq(data, model):
    return data[[model, 'freq']].groupby('freq').mean().rolling(6).mean()

def plot_average_entropy_sns(data, scatter):
    x_bins = 500
    ci = 90
    sns.regplot("freq", "type", data, logx=True, label='Type model', scatter=scatter, x_bins=x_bins, ci=ci)
    sns.regplot("freq", "token", data, logx=True, label='Token model', scatter=scatter, x_bins=x_bins, ci=ci)
    sns.regplot("freq", "two_stage", data, logx=True, label='Two-stage model', scatter=scatter, x_bins=x_bins, ci=ci)
    sns.regplot("freq", "generator", data, logx=True, label='Generator', scatter=scatter, x_bins=x_bins, ci=ci)
    plt.xscale('log')
    plt.xlabel('Word freq')
    plt.ylabel('Word neg logprob')
    plt.legend()
    plt.show()

def plot_average_entropy_sns_lm(data, scatter):
    # data wrangling 
    wrangled = defaultdict(list)
    for i, row in data.iterrows():
        for model in ['type', 'token', 'two_stage', 'generator']:
            model_entropy = row[model]
            wrangled['entropy'].append(model_entropy)
            wrangled['model'].append(model)
            wrangled['freq'].append(row['freq'])
    wrangled_data = pd.DataFrame.from_dict(wrangled)
    sns.lmplot('freq', 'entropy', wrangled_data, logx=True, hue='model', x_bins=500, scatter=scatter)
    plt.xscale('log')
    plt.xlabel('Word freq')
    plt.ylim(0, 20)
    #plt.ylabel('Word neg logprob')
    #plt.legend()
    plt.show()

def plot_original_data(sorted_data, x_axis_val):
    plt.plot(sorted_data[x_axis_val], sorted_data['type'], label='Type model')
    plt.plot(sorted_data[x_axis_val], sorted_data['token'], label='Token model')
    plt.plot(sorted_data[x_axis_val], sorted_data['two_stage'], label='Two-stage model')
    plt.plot(sorted_data[x_axis_val], sorted_data['generator'], label='Generator')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Word ' + x_axis_val)
    plt.ylabel('Word surprisal (nats)')
    plt.legend()
    plt.show()

def plot_data(x_axis, sorted_data, transform, x_label, scatter=True):
    type_color = 'b'
    token_color = 'g'
    two_stage_color = 'r'
    generator_color = '#D68910'
    if scatter:
        size = 0.1
        x_values = sorted_data['freq']+3
        jitter_lim = 0.1
        plt.scatter(x_values + np.random.uniform(-jitter_lim, jitter_lim, len(x_values)), sorted_data['type'], s=size, c=type_color)
        plt.scatter(x_values + np.random.uniform(-jitter_lim, jitter_lim, len(x_values)), sorted_data['token'], s=size, c=token_color)
        plt.scatter(x_values + np.random.uniform(-jitter_lim, jitter_lim, len(x_values)), sorted_data['generator'], s=size, c=generator_color)
        plt.scatter(x_values + np.random.uniform(-jitter_lim, jitter_lim, len(x_values)), sorted_data['two_stage'], s=size, c=two_stage_color)
    plt.plot(x_axis, transform(sorted_data, 'type'), label='Type model', c=type_color)
    plt.plot(x_axis, transform(sorted_data, 'token'), label='Token model', c=token_color)
    plt.plot(x_axis, transform(sorted_data, 'generator'), label='Generator', c=generator_color)
    plt.plot(x_axis, transform(sorted_data, 'two_stage'), label='Two-stage model', c=two_stage_color)
    plt.xscale('log')
    plt.tight_layout()
    plt.xlabel(x_label)
    plt.ylabel('Word surprisal (nats)')
    plt.ylim(4, 25)
    plt.legend()
    plt.show()

def main():
    args = get_args()
    sns.set_palette("muted")
    sns.set_context("notebook", font_scale=1.6)    
    plt.rc('font', family='serif', serif='Times New Roman')
    
    data = pd.read_csv(args.data_file)
    data['rank'] += 1
    rank_sorted = data.sort_values(by=['rank'])
    freq_sorted = data.sort_values(by=['freq'])

    plot_data(rank_sorted['rank']/2, rank_sorted, get_cumulative_average, 'Word rank', scatter=False)
    plot_data(rank_sorted['rank'].iloc[999:] - 500, rank_sorted, get_window_average, 'Word rank', scatter=False)
    plot_data(freq_sorted['freq'].unique()-3, freq_sorted, get_average_per_freq, 'Word frequency')
    #plot_original_data(sorted_data, x_axis_val)
    #plot_average_entropy_sns(data, True)
    #plot_average_entropy_sns_lm(data, False)
    #plot_average_entropy_sns(data, False)

if __name__ == '__main__':
    main()
