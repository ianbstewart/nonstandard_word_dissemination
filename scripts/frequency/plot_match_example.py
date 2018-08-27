"""
Plot example of matching growth and growth-decline word 
on s-k:s months.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import smooth_stats
from itertools import cycle
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('main_word')
    parser.add_argument('split_point', type=int)
    parser.add_argument('tf_file')
    parser.add_argument('match_words', nargs='+')
    parser.add_argument('--match_k', nargs='+', default=None)
    parser.add_argument('--out_dir', default='../../output')
    args = parser.parse_args()
    main_word = args.main_word
    match_words = args.match_words
    split_point = args.split_point
    tf_file = args.tf_file
    match_k = args.match_k
    out_dir = args.out_dir
    
    # load data
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    # tf = pd.np.log10(smooth_stats(tf))
    time_labels = sorted(tf.columns)
    T = tf.shape[1]
    time_ticks = pd.np.arange(T)
    # set up plot constants
    tick_size = 16
    label_size = 20
    legend_size = 16
    match_k = map(int, match_k)
    match_point_size = 240
    match_point_color = 'k'
    xlabel = 'Date'
    ylabel = '$\log(f)$'
    main_word_linestyle = '-'
    match_word_linestyles = cycle(['--', '-.', ':'])
    main_word_color = 'r'
    match_word_colors = cycle(['b', 'g', 'c', 'y', 'p'])
    split_color = 'k'
    split_linestyle = '-.'
#     match_point_circle_size = 200.
    
    # set x limits
    x_buffer = 12
    xmin = max(min(time_ticks), split_point - max(match_k) - x_buffer)
    xmax = min(max(time_ticks), split_point + x_buffer)
    plt.xlim((xmin, xmax))
    
    # set y limits (need ymin/ymax for vertical lines)
    all_words = [main_word] + match_words
    ymin = min(tf.loc[all_words, :].iloc[:, xmin:xmax].min(axis=1))
    ymax = max(tf.loc[all_words, :].iloc[:, xmin:xmax].max(axis=1))
    plt.ylim((ymin, ymax))

    # plot words
    line_handles = []
    line_labels = []
    l, = plt.plot(time_ticks, tf.loc[main_word], color=main_word_color, linestyle=main_word_linestyle, label=main_word)
    line_handles.append(l)
    line_labels.append(main_word)
    ax = plt.gca()
    for match_word, match_k_ in zip(match_words, match_k):
        print('plotting word %s'%(match_word))
        color_m = next(match_word_colors)
        linestyle_m = next(match_word_linestyles)
        l, = plt.plot(time_ticks, tf.loc[match_word], color=color_m, linestyle=linestyle_m, label=match_word)
        line_handles.append(l)
        line_labels.append(match_word)
        # vline at match point
        match_point_x = split_point - match_k_
#         plt.vlines(s_k, ymin, ymax, color=split_color, linestyle=split_linestyle)
        # dot and cross-hairs at match point
        match_point_y = tf.loc[[match_word], :].iloc[:, match_point_x]
        plt.scatter([match_point_x], [match_point_y], color='None', marker='o', s=match_point_size, zorder=3, edgecolor=match_point_color)
        plt.scatter([match_point_x], [match_point_y], color=match_point_color, marker='+', s=match_point_size, zorder=3)
        # plot filled circle
        # plt.plot([match_point_x], [match_point_y], ms=5, zorder=3, markeredgecolor=color_m, markerfacecolor='None')
        # plot cross-hairs: they're too thick!!
#         plt.scatter([match_point_x], [match_point_y], marker='$\\bigoplus$', color=match_point_circle_color, 
#                     s=match_point_circle_size, linestyle='None', zorder=4)
        # circle does not work!! unless you can warp the coordinates
#         match_point_circle = plt.Circle((match_point_x, match_point_y), match_point_circle_radius, color=match_point_circle_color, fill=True)
#         ax.add_artist(match_point_circle)
    
    plt.legend(line_handles, line_labels, loc='lower right', prop={'size' : legend_size})

    # add vertical line at split point
    plt.vlines(split_point, ymin, ymax, color=split_color, linestyle=split_linestyle)
    # old: vertical lines and annotation at match points
#     # x_annotate_offset = 1.
#     x_annotate_offset = 0.
#     y_annotate = 1.005*ymin
#     for k in match_k:
#         s_k = split_point - k
#         plt.annotate('s-%d'%(k), xy=[s_k - x_annotate_offset, y_annotate], 
#                      xytext=[s_k - x_annotate_offset, y_annotate])
#         plt.vlines(s_k, ymin, ymax, color=split_color, linestyle=split_linestyle)

#     plt.ylim(ymin, ymax)

    time_ticks = time_ticks[xmin:xmax]
    time_labels = time_labels[xmin:xmax]
    time_interval = 6
    time_ticks, time_labels = zip(*zip(time_ticks, time_labels)[::time_interval])
    plt.xticks(time_ticks, time_labels, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    
    plt.tight_layout()
    out_file = os.path.join(out_dir, 'match_time_series_example.png')
    plt.savefig(out_file)

if __name__ == '__main__':
    main()
