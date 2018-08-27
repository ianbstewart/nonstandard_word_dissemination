"""
Plot accuracy output from binary classification
as a bar graph of averages +/- standard deviations.
"""
from __future__ import division
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('accuracy_scores')
    parser.add_argument('--out_dir', default='../../output')
    args = parser.parse_args()
    accuracy_file = args.accuracy_scores
    out_dir = args.out_dir
    out_file = accuracy_file.replace('.tsv', '.png')
    
    # load data, aggregate
    accuracy_scores = pd.read_csv(accuracy_file, sep='\t', index_col=0)
    folds = accuracy_scores.loc[:, 'folds'].unique()[0]
    accuracy_scores.loc[:, 'AccuracySD'] = accuracy_scores.loc[:, 'AccuracySD'] * 100 / (folds)**.5
    accuracy_scores.loc[:, 'Accuracy'] = accuracy_scores.loc[:, 'Accuracy'] * 100
    
    # aggregate and sort by accuracy
    accuracy_means = accuracy_scores.loc[:, ['Accuracy', 'AccuracySD', 'feature_set_name']].groupby(by='feature_set_name').apply(pd.np.mean)
    accuracy_means = accuracy_means.sort_values('Accuracy', inplace=False, ascending=True)
    
    # plot!
    # TACL size
#     tick_size = 14
    # NWAV size
    tick_size = 20
    # TACL size
#     label_size = 20
    # NWAV size
    label_size = 28
    annotate_font_size = 12
    xlabel = 'Feature set'
    ylabel = 'Accuracy'
    # bar_color = 'b'
    bar_color = (117, 117, 255) # light blue
    bar_color = (c/255 for c in bar_color)
    bar_edge_color = 'k'
    error_color = 'k'
    feature_set_names = accuracy_means.index.tolist()
    feature_set_names_script = map(lambda n: '$\mathtt{%s}$'%(n), feature_set_names)
    feature_set_count = len(feature_set_names)
    bar_x = pd.np.arange(1, feature_set_count+1)
    bar_y = accuracy_means.loc[:, 'Accuracy']
    y_err = accuracy_means.loc[:, 'AccuracySD']
    x_offset = 0.18
    y_offset = 2.2
    error_capsize = 5
    annotate_x = bar_x - x_offset
    annotate_y = bar_y + y_offset
    plt.bar(bar_x, bar_y, color='None', edgecolor=bar_edge_color)
    # add error bars
    plt.errorbar(bar_x, bar_y, yerr=y_err, fmt='none', ecolor=error_color, capsize=error_capsize)
    # add mean annotation
    for x,y,accuracy in zip(annotate_x, annotate_y, bar_y):
        accuracy_str = '%.2f'%(accuracy)
        plt.annotate(accuracy_str, xy=(x,y), xytext=(x,y), fontsize=annotate_font_size)
    # add labels
    plt.xticks(bar_x, feature_set_names_script, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.tight_layout()
    plt.savefig(out_file)
    
if __name__ == '__main__':
    main()
