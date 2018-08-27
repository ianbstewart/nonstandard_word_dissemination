"""
Plot example growth-decline words 
with piecewise and logistic distribution
fits.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.vis_helper import plot_logistic_fit, plot_piecewise_fit
from data_processing.data_handler import smooth_stats, get_default_vocab
from argparse import ArgumentParser
import os
import seaborn as sns
sns.set_style('white')
sns.set(style='ticks', context='paper')

def main():
    parser = ArgumentParser()
    parser.add_argument('--piecewise_word', default='wot')
    parser.add_argument('--logistic_word', default='iifym')
    # parser.add_argument('--tf_file', default='../../data/frequency/2013_2016_tf_norm.tsv')
    parser.add_argument('--tf_file', default='../../data/frequency/2013_2016_tf_norm_log.tsv')
    parser.add_argument('--piecewise_params', default='../../data/frequency/2013_2016_tf_norm_2_piecewise.tsv')
    parser.add_argument('--logistic_params', default='../../data/frequency/2013_2016_tf_norm_logistic_params.tsv')
    parser.add_argument('--out_dir', default='../output/')
    args = parser.parse_args()
    piecewise_word = args.piecewise_word
    logistic_word = args.logistic_word
    tf_file = args.tf_file
    piecewise_params = args.piecewise_params
    logistic_params = args.logistic_params
    out_dir = args.out_dir
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    # vocab = get_default_vocab()
    # tf = pd.np.log10(smooth_stats(tf.loc[vocab, :].fillna(0, inplace=False)))
    piecewise_params = pd.read_csv(piecewise_params, sep='\t', index_col=0)
    logistic_params = pd.read_csv(logistic_params, sep='\t', index_col=0)
    xlabel_count = 4
    # # first piecewise
    # x0, y0, m1, m2 = piecewise_params.loc[piecewise_word, ['x0', 'y0', 'k1', 'k2']]
    # tf_w = tf.loc[piecewise_word, :]
    # out_name = 'growth_decline_piecewise_example.png'
    # out_file = os.path.join(out_dir, out_name)
    # plot_piecewise_fit(piecewise_word, tf_w, x0, y0, m1, m2, xlabel_count=xlabel_count, out_file=out_file)
    # # then logistic
    # tf_w = tf.loc[logistic_word, :]
    # loc_w = logistic_params.loc[logistic_word, 'loc']
    # scale_w = logistic_params.loc[logistic_word, 'scale']
    # out_name = 'growth_decline_logistic_example.png'
    # out_file = os.path.join(out_dir, out_name)
    # plot_logistic_fit(logistic_word, tf_w, loc_w, scale_w, xlabel_count=xlabel_count, out_file=out_file)
    # combined piecewise and logistic
    plot_count = 2
#     f, axs = plt.subplots(plot_count, sharex=True)
    f, axs = plt.subplots(1, plot_count, sharex=False, sharey=False)
    plot_width = 5.5
    plot_height = 3
    fig_width = plot_width * plot_count
    fig_height = plot_height
    f.set_size_inches(fig_width, h=fig_height)
    xlabel = 'Date'
    ylabel = 'f'
    label_size = 18
    # first piecewise
    ax1 = axs[0]
    x0, y0, m1, m2 = piecewise_params.loc[piecewise_word, ['x0', 'y0', 'k1', 'k2']]
    tf_w = tf.loc[piecewise_word, :]
    piecewise_legend_loc = 'upper left'
    plot_piecewise_fit(piecewise_word, tf_w, x0, y0, m1, m2, xlabel_count=xlabel_count, legend_loc=piecewise_legend_loc, ax=ax1)
    # add x, y labels
    ax1.set_xlabel(xlabel, fontsize=label_size)
    ax1.set_ylabel(ylabel, fontsize=label_size)
    # then logistic
    ax2 = axs[1]
    tf_w = tf.loc[logistic_word, :]
    loc_w = logistic_params.loc[logistic_word, 'loc']
    scale_w = logistic_params.loc[logistic_word, 'scale']
    logistic_legend_loc = 'upper left'
    plot_logistic_fit(logistic_word, tf_w, loc_w, scale_w, xlabel_count=xlabel_count, legend_loc=logistic_legend_loc, ax=ax2)
    # add x label
    ax2.set_xlabel(xlabel, fontsize=label_size)
    # squeeze it all in
    plt.tight_layout()
    # write to file
    out_name = 'growth_decline_piecewise_logistic_example.pdf'
    out_file = os.path.join(out_dir, out_name)
    plt.savefig(out_file)

if __name__ == '__main__':
    main()
