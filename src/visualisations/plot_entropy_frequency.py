import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.append('./src/')
from util.argparser import get_argparser, parse_args

def get_args():
    argparser = get_argparser()
    argparser.add_argument('--data-file', type=str, required=True)
    args = parse_args(argparser)
    return args

def main():
    args = get_args()
    sns.set_palette("muted")
    sns.set_context("notebook", font_scale=1.5)    
    plt.rc('font', family='serif', serif='Times New Roman')
    
    data = pd.read_csv(args.data_file)
    sorted_data = data.sort_values(by=['rank'])
    sorted_data = sorted_data.iloc[::800, :]
    plt.plot(sorted_data['rank'], sorted_data['type_loss'], label='Type model')
    plt.plot(sorted_data['rank'], sorted_data['token_loss'], label='Token model')
    plt.plot(sorted_data['rank'], sorted_data['two_stage_entr'], label='Two-stage model')
    plt.ylim(0, 25)
    plt.xlim(0, 20000)
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    main()
