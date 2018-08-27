"""
Helper functions to generate cool visualizations
to help explain the data.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# BE CAREFUL importing seaborn can change default format of plots
import seaborn as sns
from itertools import cycle, izip
from math import ceil, floor
from scipy.stats import logistic
# TODO: how to use this import statement from outside files??
# from data_handler import piecewise_linear

def plot_comparative_stat_hists(stats, stat_names, groups, group_names, out_file=None, bins=20, normed=False):
    """
    Plot histograms to compare stats across 
    different groups in same plot.
    
    Parameters:
    -----------
    stats : [pandas.DataFrame]
    stat_names : [str]
    groups : [[str]]
    List of word lists, each containing a different group.
    group_names : [str]
    out_file : str
    bins : int
    normed : bool
    """
    cols = 3
    rows = int(len(stats) / cols) + 1
    size = 4
    plt.figure(figsize=(cols * size, rows * size))
    for i, (stat, stat_name) in enumerate(izip(stats, stat_names)):
        plt.subplot(rows, cols, i+1)
        # TODO: more colors
        colors = cycle([(0,0,1.0,0.5), (1.0,0,0,0.5), (0,1.0,0,0.5)])
        # get common bins for all groups
        combined_stats = pd.concat([stat.loc[group] for group in groups])
        _, combined_bin_edges = pd.np.histogram(combined_stats, bins=bins)
        for group, group_name in izip(groups, group_names):
            group_vals = stat.loc[group]
            n, bin_edges = pd.np.histogram(group_vals, bins=combined_bin_edges)
            if(normed):
                n = n / n.sum()
            bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
            bin_width = (bin_edges[1] - bin_edges[0])
            plt.bar(bin_mids, n, width=bin_width, fc=next(colors), label=group_name)
        plt.legend(loc='upper left')
        plt.title('%s distribution'%(stat_name))
    if(out_file is not None):
        plt.savefig(out_file)
    else:
        plt.show()

def plot_words(w1, w2, stat, xtick_ctr=4):
    N = stat.shape[1]
    X = range(N)
    xlabels = sorted(stat.columns)
    xtick_space = int(N / xtick_ctr)
    xticks, xlabels = zip(*zip(X, xlabels)[::xtick_space])
    plt.plot(X, stat.loc[w1], 'b', label=w1)
    plt.plot(X, stat.loc[w2], 'r', label=w2)
    plt.legend(loc='upper left')
    plt.xticks(xticks, xlabels)

def plot_all_word_pairs(words1, words2, stat, out_file=None):
    """
    Plot word pair time series, one per subplot.

    Parameters:
    -----------
    words1 : [str]
    words2 : [str]
    stat : pandas.DataFrame
    out_file : str
    """
    all_words = words1 + words2
    N = len(words1)
    cols = 4
    rows = int(N / cols) + 1
    size = 4
    plt.figure(figsize=(cols * size, rows * size))
    ctr = 1
    ylim = (stat.loc[all_words].min().min(), stat.loc[all_words].max().max())
    for w1, w2 in izip(words1, words2):
        plt.subplot(rows, cols, ctr)
        plot_words(w1, w2, stat)
        plt.ylim(ylim)
        ctr += 1
    plt.tight_layout()
    if(out_file is not None):
        plt.savefig(out_file)

def plot_word_series(stat, words, out_file=None, break_points=None, suptitle=None, x_tick_count=6):
    """
    Plot word stat time series in separate subplots.

    Parameters:
    -----------
    stat : pandas.DataFrame
    Rows = words, cols = dates.
    words : [str]
    out_file : str
    break_points : [int]
    Timesteps at which to draw vertical red lines.
    suptitle : str
    x_tick_count : int
    Number of x ticks.
    """
    cols = 3
    rows = int(len(words) / cols) + 1
    size = 4
    xlabels = stat.columns
    xticks = range(stat.shape[1])
    D = int(ceil(len(xlabels) / x_tick_count)) + 1
    xlabels, xticks = zip(*zip(xlabels, xticks)[::D])
    plt.figure(figsize=(cols * size, rows * size))
    X = range(stat.shape[1])
    for i, word in enumerate(words):
        stat_w = stat.loc[word]
        plt.subplot(rows, cols, i+1)
        plt.plot(X, stat_w)
        plt.title(word, fontsize=18)
        plt.xticks(xticks, xlabels)
        # plot optional lines to specify regions of interest
        if(break_points is not None):
            ymin = min(stat_w)
            ymax = max(stat_w)
            for k in break_points:
                plt.plot((k, k), (ymin, ymax), 'r')
    plt.tight_layout()
    if(suptitle is not None):
        plt.suptitle(suptitle, y=1.08, fontsize=48)
    if(out_file is not None):
        plt.savefig(out_file)
    else:
        plt.show()

def plot_separate_stats(word, stat1, stat2, stat1_name, stat2_name, ax=None, out_file=None, xtick_ctr=8, ylims=None):
    """
    Plot same word, different stats on same x axis.
    
    Parameters:
    -----------
    word : str
    stat1 : pandas.Series
    # index = dates
    stat2 : pandas.Series
    stat1_name : str
    stat2_name : str
    ax : pyplot.Axes
    Optional axis object, in case of subplots.
    out_file : str
    Optional output file name.
    xtick_ctr : int
    Optional number of x ticks to include in plot.
    ylims : [(float, float)]
    Optional limit for both y axes.
    """
    N = len(stat1)
    X = range(N)
    x_labels = stat1.index
    xtick_interval = int(ceil(N / xtick_ctr)) + 1
    x_ticks, x_labels = zip(*zip(X, x_labels)[::xtick_interval])
    if(ax is None):
        ax1 = plt.gca()
    else:
        ax1 = ax
    ax2 = ax1.twinx()
    l1 = ax1.plot(X, stat1, color='r', label=stat1_name)
    l2 = ax2.plot(X, stat2, color='b', label=stat2_name)
    ax1.set_ylabel(stat1_name, fontsize=24)
    ax2.set_ylabel(stat2_name, fontsize=24)
    lines = l1 + l2
    labels = [stat1_name, stat2_name]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title(word, fontsize=24)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    if(ylims is not None):
        ax1.set_ylim(ylims[0])
        ax2.set_ylim(ylims[1])
    if(out_file is not None):
        plt.savefig(out_file)

def plot_word_curve(w, stat, stat_fit, stat_name, out_file=None):
    """
    Plot word stat as scatterplot 
    and the corresponding fit stat
    as a smooth curve.
    
    Parameters:
    -----------
    w : str
    stat : pandas.Series
    stat_fit : pandas.Series
    stat_name : str
    out_file : str
    """
    stat_color = 'r'
    curve_color = 'b'
    X = pd.np.arange(len(stat))
    xticks = pd.np.arange(len(stat))
    xlabels = sorted(stat.index.tolist())
    xtick_interval = 8
    xticks, xlabels = zip(*zip(xticks, xlabels)[::xtick_interval])
    plt.scatter(X, stat, color=stat_color)
    plt.plot(X, stat_fit, color=curve_color)
    plt.xticks(xticks, xlabels)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(stat_name, fontsize=18)
    plt.title(w, fontsize=24)
    if(out_file is not None):
        plt.savefig(out_file)

def plot_piecewise(word, stat, stat_name, params, out_file=None):
    """
    Plot word stats as scatterplot and the
    linear piecewise function as a line plot.
    
    Parameters:
    -----------
    word : str
    stat : pandas.Series
    stat_name : str
    params : list
    Parameter list containing: 
    x0 (switch point x position), 
    y0 (switch point y position), 
    k1 (slope of first line), 
    k2 (slope of second line). 
    """
    scatter_color = 'b'
    func_color = 'r'
    X = pd.np.arange(len(stat), dtype=float)
    xticks = X.copy()
    xlabels = sorted(stat.index.tolist())
    xtick_interval = 8
    xticks, xlabels = zip(*zip(xticks, xlabels)[::xtick_interval])
    y = stat.copy()
    plt.scatter(X, y, color=scatter_color)
    y_fit = piecewise_linear(X, *params)
    plt.plot(X, y_fit, color=func_color)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(stat_name, fontsize=18)
    plt.title(word, fontsize=24)
    plt.xticks(xticks, xlabels)
    if(out_file is not None):
        plt.savefig(out_file)

def plot_jointplot(stats, x_stat_name, y_stat_name, size=5, xlim=None, ylim=None, annotate_words=None, bins='log', out_file=None):
    """
    Produce a joint plot with distribution of x stat and y stat,
    using a hex plot to approximate the distribution density. 
    NOT PERFECT BUT IT WORKS.

    Parameters:
    -----------
    stats : pandas.DataFrame
    Rows = words, columns = stats.
    x_stat_name : str
    y_stat_name : str
    size : int
    Optional size in inches (square).
    xlim : (float, float)
    Optional x limits.
    ylim : (float, float)
    Optional y limits.
    annotate_words : [str]
    Optional words to be plotted with the distribution. Gets crowded quickly!
    bins : int
    Optional number of bins to use in plot. Default = "log", automatically determining bins based on log scale.
    out_file : str
    Optional file to write plot.
    """
    annotate_font = 8
    label_font = 24
    title_font = 32
    bin_color = 'g'
    cmap = plt.get_cmap('Greens')
    g = sns.JointGrid(x=x_stat_name, y=y_stat_name, data=stats, xlim=xlim, ylim=ylim)
    g.plot_marginals(sns.distplot, color=bin_color)
    g.plot_joint(plt.hexbin, cmap=cmap, bins=bins)
    if(annotate_words is not None):
        for w in annotate_words:
            x, y = (stats.loc[w, x_stat_name], stats.loc[w, y_stat_name])
            plt.annotate(w, xy=(x,y))
    y_name_tex = '$%s$'%(y_stat_name)
    title = '%s distribution'%(y_name_tex)
    plt.xlabel(x_stat_name, fontsize=label_font)
    plt.ylabel(y_name_tex, fontsize=label_font)
    plt.suptitle(title, fontsize=title_font, y=1.08)
    if(out_file is not None):
        plt.savefig(out_file)

def plot_piecewise_fit(w, tf, x0, y0, m1, m2, xlabel_count=4, legend_loc='upper right', out_file=None, ax=None):
    """
    Plot time series for given word and the
    best-fit two-part piecewise function.
    
    Parameters:
    -----------
    w : str
    tf : pandas.Series
    x0 : float
    X-coordinate of split point.
    y0 : float
    Y-coordinate of split point.
    m1 : float
    Slope of first line.
    m2 : float
    Slope of second line.
    legend_loc : str
    Legend location.
    xlabel_count : int
    Number of xlabels to include.
    out_file : str
    ax : matplotlib.axes.Axes
    """
    label_font = 18
    title_font = 24
    tick_size = 14
    legend_size = 14
    b1 = y0 - m1*x0
    b2 = y0 - m2*x0
    N = len(tf)
    X = pd.np.arange(N)
    xlabels = sorted(tf.index)
    xlabel_interval = int(ceil(N / (xlabel_count))) + 1
    xticks, xlabels = zip(*zip(X, xlabels)[::xlabel_interval])
    xlabel = 'Date'
    ylabel = 'log(f)'
    # compute fit lines
    s = int(ceil(x0))
    fit_line_y1 = m1*X[:s] + b1
    fit_line_y2 = m2*X[s:] + b2
    fit_line_y = pd.np.concatenate([fit_line_y1, fit_line_y2])
    series_color = 'r'
    fit_color = 'b'
    series_linestyle = '-'
    fit_linestyle = '--'
    split_color = 'k'
    split_linestyle = '--'
    single_axis = ax is None
    if(single_axis):
        plt.figure(figsize=(5,5))
        ax = plt.subplot(111)
    # first draw actual values, then fit line
    l1, = ax.plot(X, tf, color=series_color, linestyle=series_linestyle)
    l2, = ax.plot(X, fit_line_y, color=fit_color, linestyle=fit_linestyle)
    # add dotted line for split point
#     ylim = (min(tf.min(), fit_line_y.min()), max(tf.max(), fit_line_y.max()))
#     ax.plot([s, s], ylim, color=split_color, linestyle=split_linestyle)
    # set legend series line and fit line
    lines = [l1, l2]
#     labels = [w, 'piecewise_fit']
    labels = ['observed', 'piecewise fit']
    ax.legend(lines, labels, fontsize=legend_size, loc=legend_loc)
    # fix tick size
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=tick_size)
    # force display the ticks
    plt.setp(ax.get_xticklabels(), fontsize=tick_size, visible=True)
    yticks = ax.get_yticks()
    ylabels = map(lambda t: '%.2f'%(t), yticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=tick_size)
    ax.set_title(w, fontsize=title_font)
    # if single axis, add x and labels
    if(single_axis):
        ax.set_xlabel(xlabel, fontsize=label_font)
        ax.set_ylabel(ylabel, fontsize=label_font)
    if(out_file is not None):
        plt.tight_layout()
        plt.savefig(out_file)

def plot_logistic_fit(w, tf, loc, scale, xlabel_count=4, legend_loc='upper right', out_file=None, ax=None):
    """
    Plot time series for given word and the 
    best-fit logistic distribution.
    
    Parameters:
    -----------
    w : str
    tf : pandas.Series
    loc : float
    scale : float
    xlabel_count : int
    legend_loc : str
    out_file : str
    ax : matplotlib.axes.Axes
    """
    label_font = 18
    title_font = 24
    tick_size = 14
    legend_size = 14
    N = len(tf)
    X = pd.np.arange(N)
    xlabels = sorted(tf.index)
    xlabel_interval = int(ceil(N / (xlabel_count))) + 1
    xticks, xlabels = zip(*zip(X, xlabels)[::xlabel_interval])
    xlabel = 'Date'
    ylabel = 'log(f)'
    logistic_y = logistic.pdf(X, loc=loc, scale=scale)
    # rescale logistic y to match tf: 
    # y_logistic_rescaled = y_logistic * y_sum + y_offset
    y_offset = tf.min()
    tf_rescaled = tf - y_offset
    logistic_y_rescaled = logistic_y * tf_rescaled.sum() + y_offset
    series_color = 'r'
    fit_color = 'b'
    series_linestyle = '-'
    fit_linestyle = '--'
    split_color = 'k'
    split_linestyle = '--'
    single_axis = ax is None
    if(single_axis):
        plt.figure(figsize=(5,5))
        ax = plt.subplot(111)
    l1, = ax.plot(X, tf, color=series_color, linestyle=series_linestyle)
    l2, = ax.plot(X, logistic_y_rescaled, color=fit_color, linestyle=fit_linestyle)
    # add legend
    lines = [l1, l2]
#     labels = [w, 'logistic_fit']
    labels = ['observed', 'logistic fit']
    ax.legend(lines, labels, fontsize=legend_size, loc=legend_loc)
    # add dotted line for split point
#     ylim = ax.get_ylim()
#     ax.plot([loc, loc], ylim, color=split_color, linestyle=split_linestyle)
    # set ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=tick_size)
    yticks = ax.get_yticks()
    ylabels = map(lambda t: '%.2f'%(t), yticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=tick_size)
    ax.set_title(w, fontsize=title_font)
    # if single axis, add x and labels
    if(single_axis):
        ax.set_xlabel(xlabel, fontsize=label_font)
        ax.set_ylabel(ylabel, fontsize=label_font)
    if(out_file is not None):
        plt.tight_layout()
        plt.savefig(out_file)

def set_color(bp, color, linestyle='-'):
    """
    Set color of boxplot.
    
    Parameters:
    -----------
    bp : matplotlib
    color : str
    """
    plt.setp(bp['boxes'], color=color, linestyle=linestyle)
    plt.setp(bp['caps'], color=color, linestyle=linestyle)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle)
    plt.setp(bp['medians'], color=color, linestyle=linestyle)

def compare_boxplots(boxdata_1, boxdata_2, x_labels, x_title, y_title, name_1, name_2, 
                     color_1='r', color_2='b', linestyle_1='-', linestyle_2='-.', 
                     label_size=18, tick_size=15, ylim=None, out_file=None):
    """
    Plot two sets of boxplot data on same grid, but offset
    slightly to compare side-by-side.
    
    Parameters:
    -----------
    boxdata_1 : [pandas.Series]
    boxdata_2 : [pandas.Series]
    x_labels : [str]
    x_title : str
    y_title : str
    name_1 : str
    name_2 : str
    color_1 : str
    color_2 : str
    linestyle_1 : str
    linestyle_2 : str
    label_size : int
    tick_size : int
    ylim : (float, float)
    out_file : str
    """
    legend_size = 15
    N = len(x_labels)
    x_width = 1.5
    box_width = x_width / 4.
    height = 5
    plt.figure(figsize=(x_width * N, height))
    x_offset = 0.5
    x_positions = pd.np.arange(N)
    bp_1 = plt.boxplot(boxdata_1, positions=x_positions, widths=box_width)
    bp_2 = plt.boxplot(boxdata_2, positions=x_positions+x_offset, widths=box_width)
    set_color(bp_1, color_1, linestyle=linestyle_1)
    set_color(bp_2, color_2, linestyle=linestyle_2)
    plt.xticks(x_positions + x_offset / 2., x_labels, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlim((x_positions[0]-1, x_positions[-1] + 1))
    plt.xlabel(x_title, fontsize=label_size)
    plt.ylabel(y_title, fontsize=label_size)
    # dummy lines for legend
    l1, = plt.plot([0,0], [1,1], color=color_1, label=name_1, linestyle=linestyle_1)
    l2, = plt.plot([0,0], [1,1], color=color_2, label=name_2, linestyle=linestyle_2)
    plt.legend(loc='lower right', prop={'size' : legend_size})
    if(ylim is not None):
        plt.ylim(ylim)
    if(out_file is not None):
        plt.savefig(out_file)
