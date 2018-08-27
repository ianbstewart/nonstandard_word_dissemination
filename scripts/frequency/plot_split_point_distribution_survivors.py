"""
Plot survivor proportion curve.
"""
from __future__ import division
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_growth_decline_words_and_params, get_growth_words
from math import ceil
import os
from argparse import ArgumentParser
from datetime import datetime
from dateutil.relativedelta import relativedelta

def main():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='../../output')
    args = parser.parse_args()
    out_dir = args.out_dir
    
    # load data
    growth_decline_words, split_points = get_growth_decline_words_and_params()
    split_points = split_points.apply(lambda x: int(ceil(x)))
    # drop bad split points
    T = 36
    split_points = split_points[(split_points > 0) & (split_points < T)]
    growth_words = get_growth_words()
    GD = len(growth_decline_words)
    G = len(growth_words)
    N = G + GD
    survivors = pd.np.repeat(N, T)
    deaths = pd.Series(pd.np.zeros(T), index=pd.np.arange(T))
    deaths = (deaths + split_points.value_counts()).fillna(0, inplace=False)
    deaths_cumulative = deaths.cumsum()
    survivors -= deaths_cumulative
    timesteps = pd.np.arange(T)
    t_0 = '2013-06'
    t_0 = datetime.strptime(t_0, '%Y-%m')
    time_labels = [datetime.strftime(t_0 + relativedelta(months=+d), '%Y-%m') for d in range(T)]
    time_interval = 8
    time_ticks, time_labels = zip(*zip(timesteps, time_labels)[::time_interval])

    # make curve
    x_buffer = 0.5
    y_buffer = 50
    xlabel = 'Date'
    ylabel = 'Survivors'
    label_size = 20
    tick_size = 14
    survivor_marker_size = 10
    survivor_color = 'k'
    survivor_linestyle = '-'
    fill_hatch = '//'
    # fill_color = 'b'
    # use light-blue as fill color
    fill_color = (117, 117, 255)
    fill_color = tuple(c/255 for c in fill_color)
    xlim = [min(timesteps)-x_buffer, max(timesteps)+x_buffer]
    # cutoff at y=0
    # ylim = [min(survivors) - y_buffer, max(survivors)+y_buffer]
    ylim = [0, max(survivors)+y_buffer]
    plt.plot(timesteps, survivors, color=survivor_color, linestyle=survivor_linestyle, zorder=2)
    # add markers
    plt.scatter(timesteps, survivors, color=survivor_color, s=survivor_marker_size, zorder=3)
    # add dotted line at lower bound
    lower_bound_x = [0 - x_buffer, max(timesteps) + x_buffer]
    lower_bound_y = [G, G]
    plt.plot(lower_bound_x, lower_bound_y, color='k',  linestyle='--')
    # fill between survivor curve and lower bound
    plt.fill_between(timesteps, survivors, facecolor='none', hatch=fill_hatch, edgecolor=fill_color, linewidth=0.0)
#     plt.fill_between(timesteps, survivors, hatch='X', edgecolor='none', facecolor=fill_color, zorder=1)
    # fix ticks
    plt.xticks(time_ticks, time_labels, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    # add bracket annotation for growth/failure
    def bracket_text(x, y1, y2, text, fraction=0.2, text_x_offset=2., text_y_offset=20):
        connection_style = 'bar, fraction=%.2f'%(fraction)
        plt.annotate('', xy=(x,y1), xycoords='data', xytext=(x,y2), textcoords='data', arrowprops=dict(arrowstyle='-', connectionstyle=connection_style))
        text_x = x + text_x_offset
        text_y = (y1 + y2) / 2. + text_y_offset
        plt.text(text_x, text_y, text, rotation=270.)
    growth_bracket_x = max(timesteps) + .5
    # growth bracket
    growth_bracket_y1 = G * .75
    growth_bracket_y2 = G * .25
    growth_text = 'growth'
    text_y_offset = 110
    bracket_text(growth_bracket_x, growth_bracket_y1, growth_bracket_y2, growth_text, text_y_offset=text_y_offset)
    # failure bracket
    failure_bracket_x = max(timesteps) + .5
    failure_bracket_y1 = N * .95
    failure_bracket_y2 = G * 1.05
    failure_text = 'decline'
    text_y_offset = 35
    bracket_text(failure_bracket_x, failure_bracket_y1, failure_bracket_y2, failure_text, fraction=0.3, text_y_offset=text_y_offset)
    # squeeze layout
    plt.tight_layout()
    # write to file
    out_file = os.path.join(out_dir, 'split_point_survivor_curve.pdf')
    plt.savefig(out_file, bbox_inches='tight')

if __name__ == '__main__':
    main()
