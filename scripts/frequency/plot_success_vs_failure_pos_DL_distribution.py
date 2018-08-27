"""
Plot distribution of DL values across growth and growth-decline
words, grouped by POS tag. WHEW.
"""
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_growth_words, get_growth_decline_words_and_params, get_default_vocab
from prediction.prediction_helpers import match_words_split_points, match_word_diffs_all_pairs
from data_processing.vis_helper import compare_boxplots
from math import ceil
from scipy.stats import ttest_ind
from argparse import ArgumentParser
import seaborn as sns
sns.set_style('white')

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/frequency')
    parser.add_argument('--match_stat', default='../../data/frequency/2013_2016_tf_norm_log.tsv')
    parser.add_argument('--plot_stat', default='../../data/frequency/2013_2016_3gram_residuals.tsv')
    parser.add_argument('--tag_pcts', default='../../data/frequency/2013_2016_tag_pcts.tsv')
    parser.add_argument('--out_dir', default='../../output')
    args = parser.parse_args()
    data_dir = args.data_dir
    match_stat_file = args.match_stat
    plot_stat_file = args.plot_stat
    tag_pct_file = args.tag_pcts
    out_dir = args.out_dir
    growth_words = get_growth_words()
    decline_words, split_points = get_growth_decline_words_and_params()
    split_points = split_points.apply(lambda x: int(ceil(x)))
    
    vocab = get_default_vocab()
    # match_stat = pd.read_csv(os.path.join(data_dir, '2013_2016_tf_norm.tsv'), sep='\t', index_col=0).loc[vocab, :]
    match_stat = pd.read_csv(match_stat_file, sep='\t', index_col=0)
    DL = pd.read_csv(plot_stat_file, sep='\t', index_col=0)
    min_diff_pct = 0
    # match on split point
#     k = 1 
#     match_diffs = match_words_split_points(decline_words, growth_words, match_stat, split_points, k, min_diff_pct, replace=False)
    # match on first k months of data
    k = 12
    match_diffs = match_word_diffs_all_pairs(decline_words, growth_words, match_stat, k, min_diff_pct=min_diff_pct)
    
    # tag_estimates = pd.read_csv(os.path.join(data_dir, '2013_2016_tag_pcts.tsv'), sep='\t', index_col=0).apply(lambda x: x.argmax(), axis=1)
    # use tag estimates without proper nouns
    tag_estimates = pd.read_csv(tag_pct_file, sep='\t', index_col=0).drop('^', inplace=False, axis=1).apply(lambda x: x.argmax(), axis=1)
    decline_words_matched = match_diffs.loc[:, 'word'].tolist()
    growth_words_matched = match_diffs.loc[:, 'match'].tolist()
    split_points_ordered = split_points.loc[decline_words_matched]
    split_points_growth = pd.Series(split_points_ordered)
    split_points_growth.index = growth_words_matched
    combined_words = decline_words_matched + growth_words_matched
    tag_estimates_combined = tag_estimates.loc[combined_words]
    tag_list = []
    growth_vals = []
    decline_vals = []
    ttest_results = []
    ttest_results = pd.DataFrame()
    min_count = 5
    DL_k = DL.iloc[:, 1:k]
    for t, group in tag_estimates_combined.groupby(tag_estimates_combined):
        decline_relevant = list(group.index & set(decline_words_matched))
        growth_relevant = list(group.index & set(growth_words_matched))
        if((len(decline_relevant) >= min_count) and (len(growth_relevant) >= min_count)):
            tag_list.append(t)
            # now! get DL values
            # get mean DL values
            decline_DL = DL_k.loc[decline_relevant, :].mean(axis=1)
            growth_DL = DL_k.loc[growth_relevant, :].mean(axis=1)
            decline_vals.append(decline_DL)
            growth_vals.append(growth_DL)
    
            # t-test for significance
            tval, pval = ttest_ind(growth_DL, decline_DL, equal_var=False)
            pval /= 2 # divide by two because one-sided
            # track means, t-val, p-val
            ttest_results_ = pd.Series(
                {
                    'POS_tag' : t,
                    'growth_DL_mean' : growth_DL.mean(),
                    'growth_DL_sd' : growth_DL.std(),
                    'growth_DL_N' : len(growth_DL),
                    'growth_DL_mean' : decline_DL.mean(),
                    'growth_DL_sd' : decline_DL.std(),
                    'growth_DL_N' : len(decline_DL),
                    't' : tval,
                    'p' : pval,
                })
            ttest_results = ttest_results.append(ttest_results_, ignore_index=True)
#             ttest_results.append((t, pval))
    name_1 = 'growth'
    name_2 = 'decline'
    xlabel = 'POS tag'
    ylabel = '$D^{L}$'
    ylim = (-1., 0.5)
    # TACL size
    tick_size = 15
    # NWAV size
#     tick_size = 18
    # save ttest to file first
    ttest_out_file = os.path.join(out_dir, '%s_vs_%s_matched_pos_DL_distribution_1_%d.tsv'%(name_1, name_2, k))
    ttest_results.to_csv(ttest_out_file, sep='\t', index=False)
    out_file = os.path.join(out_dir, '%s_vs_%s_matched_pos_DL_distribution_1_%d.pdf'%(name_1, name_2, k))
    # convert tag list to meanings
    tag_meanings = pd.read_csv('../../data/metadata/tag_meaning.tsv', sep='\t', index_col=0).applymap(lambda x: x.split('/')[0].replace(' ', '\n'))#replace('/', '\n'))
    tag_list = [tag_meanings.loc[t, 'meaning'] for t in tag_list]
    # plot boxes
    color_1 = 'b'
    color_2 = 'r'
    linestyle_1 = '--'
    linestyle_2 = '-'
    # TACL size
#     label_size = 18
    # NWAV size
    label_size = 28
    compare_boxplots(growth_vals, decline_vals, tag_list, xlabel, ylabel, name_1, name_2, 
                     color_1=color_1, color_2=color_2, linestyle_1=linestyle_1, linestyle_2=linestyle_2,
                     label_size=label_size, tick_size=tick_size, ylim=ylim)

    # add xticks
    x_offset = 0.25
    x_positions = pd.np.arange(len(tag_list)) + x_offset
    plt.xticks(x_positions, tag_list, fontsize=tick_size)
    # add significance stars
    # new: add as brackets between boxes
    def bracket_text(x1_bracket, x2_bracket, y_bracket, x_txt, y_txt, text, fraction=0.2, textsize=12, bracket_color='black'):
        connection_style = 'bar, fraction=%.2f'%(fraction)
        arrowprops = dict(arrowstyle='-', ec=bracket_color, connectionstyle=connection_style)
        plt.annotate('', xy=(x1_bracket,y_bracket), xycoords='data', 
                     xytext=(x2_bracket, y_bracket), textcoords='data', arrowprops=arrowprops)
        plt.text(x_txt, y_txt, text, rotation=0., size=textsize, weight='bold')
    pval_upper = 0.05
    # ttest_results is a data frame
    x_positions_significant = [x_positions[i] for i in range(len(x_positions)) if ttest_results.iloc[i, :].loc['p'] <pval_upper]
    bracket_y = max(max(map(max, growth_vals)), max(map(max, decline_vals)))
    bracket_x_offset = 0.25
    text_x_offset = -0.025
    text_y_offset = 0.1
    fraction = 0.3
    annotate_txt = '*'
    annotate_txt_size = 15
    for x_position in x_positions_significant:
        bracket_x1 = x_position - bracket_x_offset
        bracket_x2 = x_position + bracket_x_offset
        x_txt = (bracket_x1 + bracket_x2) / 2. + text_x_offset
        y_txt = bracket_y + text_y_offset
        bracket_text(bracket_x1, bracket_x2, bracket_y, 
                     x_txt, y_txt, annotate_txt, 
                     fraction=fraction, textsize=annotate_txt_size)
    
    # update xlim to fit labels and boxes
    xmin = x_positions.min() - x_offset*2.
    xmax = x_positions.max() + x_offset*2.
    plt.xlim(xmin, xmax)
    
    plt.tight_layout()
    # remove border but keep axes
    plt.axis('on')
    # plt.box(on=False)
    plt.savefig(out_file)

if __name__ == '__main__':
    main()
