"""
Plot best-fitting growth and decline words' 
frequency time series.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_growth_words, get_growth_decline_words_and_params, get_logistic_decline_words, get_piecewise_decline_words
from math import ceil
from itertools import izip
import os

def plot_time_series(stat, words, x_lab='Month', y_lab='f', title=''):
    N = stat.shape[1]
    X = pd.np.arange(N)
    W = len(words)
    stat_color = 'blue'
    stat_linestyle = '-'
    height = 3
    plot_width = 3.
    width = W*plot_width
    label_size = 18
    title_size = 24
    sub_title_size = 16
    x_lab_offset = [0.5, 0.05]
    
    # get datetime for xticks
    xlabels = stat.columns.tolist()
    xtick_count = 4
    xlabel_interval = int(ceil(N / (xtick_count))) + 1
    xticks, xlabels = zip(*zip(X, xlabels)[::xlabel_interval])
    tick_size = 10
    suptitle_y_pos = 0.9
    
    # make subplots
    f, axs = plt.subplots(1, W, sharey=True, figsize=(width, height))
    for i, w in enumerate(words):
        ax = axs[i]
        ax.plot(X, stat.loc[w], color=stat_color, linestyle=stat_linestyle)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=tick_size)
        if(i == 0):
            ax.set_ylabel(y_lab, fontsize=label_size)
        ax.set_title(w, fontsize=sub_title_size)

    plt.suptitle(title, fontsize=title_size, y=suptitle_y_pos)
    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom_space)
    plt.figtext(x_lab_offset[0], x_lab_offset[1], x_lab, fontsize=label_size)
#     plt.text(x_lab_offset[0], x_lab_offset[1], x_lab, fontsize=label_size)
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='../../output/results/')
    parser.add_argument('--tf_file', default='../../data/frequency/2013_2016_tf_norm_log.tsv')
    parser.add_argument('--growth_score_file', default='../../data/frequency/growth_scores.tsv')
    args = parser.parse_args()
    out_dir = args.out_dir
    tf_file = args.tf_file
    growth_score_file = args.growth_score_file
    
    ## load data
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    growth_params = pd.read_csv(growth_score_file, sep='\t', index_col=0)
    growth_words = get_growth_words()
    decline_words, decline_params = get_growth_decline_words_and_params()
    logistic_decline_words, logistic_params = get_logistic_decline_words()
    piecewise_decline_words, piecewise_params = get_piecewise_decline_words()
    logistic_decline_words = list(set(logistic_decline_words) & set(decline_words) - set(growth_words))
    piecewise_decline_words = list(set(piecewise_decline_words) & set(decline_words) - set(growth_words))
    
    ## sort scores
    growth_scores = growth_params.loc[growth_words, 'spearman'].sort_values(inplace=False, ascending=False)
    decline_logistic_scores = logistic_params.loc[logistic_decline_words, 'R2'].sort_values(inplace=False, ascending=False)
    decline_piecewise_scores = piecewise_params.loc[piecewise_decline_words, 'R2'].sort_values(inplace=False, ascending=False)
    
    ## get example words
    top_k = 5
    example_growth_words = growth_scores.index.tolist()[:top_k]
    example_logistic_words = decline_logistic_scores.index.tolist()[:top_k]
    example_piecewise_words = decline_piecewise_scores.index.tolist()[:top_k]
    
    ## plot!! and write to file
    word_categories = ['growth', 'logistic_decline', 'piecewise_decline']
    word_lists = [example_growth_words, example_logistic_words, example_piecewise_words]
    for word_category, word_list in izip(word_categories, word_lists):
        plot_time_series(tf, sorted(word_list))
        out_file = os.path.join(out_dir, '%s_best_fit.pdf'%(word_category))
        # save to file
        plt.savefig(out_file)

if __name__ == '__main__':
    main()