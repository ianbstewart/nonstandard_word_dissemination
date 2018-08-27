"""
Plot frequency vs. context counts and include
some example innovations. For each innovation include
a bar measuring the residual between predicted and 
actual counts.

USED FOR NWAV 2017 POSTER
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from argparse import ArgumentParser
import os
from scipy.stats import linregress

def main():
  parser = ArgumentParser()
  parser.add_argument('tf_file', default='../../data/frequency/2013_2016_tf_norm_log.tsv')
  parser.add_argument('c3_file', default='../../data/frequency/2013_2016_unique_3gram_counts.tsv')
  parser.add_argument('example_innovations', nargs='+', 
                     default=['someshit', 'prefab', 'yikes', 'aka'])
  parser.add_argument('out_dir', default='../../output')
  args = parser.parse_args()
  tf_file = args.tf_file
  c3_file = args.c3_file
  example_innovations = args.example_innovations
  out_dir = args.out_dir
  
  ## load data
  tf = pd.read_csv(tf_file, sep='\t', index_col=0)
  c3 = pd.read_csv(c3_file, sep='\t', index_col=0)
  
  ## prepare data
  c3 = c3.applymap(lambda x: pd.np.log10(x + 1.))
  shared_idx = tf.index & c3.index
  tf_mean = tf.mean(axis=1).loc[shared_idx]
  c3_mean = c3.mean(axis=1).loc[shared_idx]
  sample_size = 10000 - len(example_innovations)
  idx_sample = list(pd.np.random.choice(shared_idx, sample_size, replace=False)) + example_innovations
  tf_sample = tf_mean.loc[idx_sample]
  c3_sample = c3_mean.loc[idx_sample]
  
  ## compute linear regression for expected counts
  m, b, r, p, err = linregress(tf_sample.values, c3_sample.values)
  c3_expected = m*tf_sample + b
  
  ## plot
  x_label = 'log(f)'
  y_label = 'log($C^{3}$)'
  label_size = 24
  tick_size = 18
  annotate_size = 18
  line_color = 'r'
  scatter_color = 'b'
  annotate_color = 'b'
  scatter_alpha = 0.05
  arrow_width = 2
  arrow_color = 'k'
  x_annotate_offset_ratio = 0.985
  width = 5
  height = 5
  x_tick_ctr = 6
  plt.figure(figsize=(width, height))
#   sns.regplot(tf_sample, c3_sample, line_kws={'color': line_color}, scatter_kws={'color' : scatter_color, 'alpha' : scatter_alpha})
  # first plot raw data and fit line
  plt.scatter(tf_sample, c3_sample, color=scatter_color, alpha=scatter_alpha)
  plt.plot(tf_sample, c3_expected, color=line_color)
  # annotate example innovations
  for i in example_innovations:
    x = float(tf_mean.loc[i])
    y = float(c3_mean.loc[i])
    y_expected = c3_expected.loc[i]
    print(type(y_expected))
    # annotations aren't customizable ;_;
    plt.annotate(i, xy=(x,y_expected), xycoords='data', xytext=(x*x_annotate_offset_ratio, y), textcoords='data', zorder=3, fontsize=annotate_size)
    y_min = min(y_expected, y)
    y_max = max(y_expected, y)
    plt.vlines(x, y_min, y_max, colors=arrow_color, linewidths=arrow_width, zorder=1)
  plt.scatter(tf_mean.loc[example_innovations], c3_mean.loc[example_innovations], color=annotate_color, zorder=2)
  
  ## restrict space to annotated points
  xlim_boundary_ratio = 0.75
  ylim_boundary_ratio = 0.75
  x_min = tf_mean.loc[example_innovations].min()
  x_max = tf_mean.loc[example_innovations].max()
  y_min = c3_mean.loc[example_innovations].min()
  y_max = c3_mean.loc[example_innovations].max()
  # want limits in ascending order
  xlim = sorted([x_min*xlim_boundary_ratio, x_max*(2-xlim_boundary_ratio)])
  ylim = sorted([y_min*xlim_boundary_ratio, y_max*(2-ylim_boundary_ratio)])
  plt.xlim(xlim)
  plt.ylim(ylim)
  plt.xlabel(x_label, fontsize=label_size)
  plt.ylabel(y_label, fontsize=label_size)
  
  ## fix ticks
  x_tick_interval = (xlim[1] - xlim[0]) / x_tick_ctr
  x_ticks = pd.np.arange(xlim[0], xlim[1], x_tick_interval)
  x_tick_labels = map(lambda x: '%.1f'%(x), x_ticks)
  plt.xticks(x_ticks, x_tick_labels, fontsize=tick_size)
  plt.yticks(fontsize=tick_size)
  
  ## save
  out_file = os.path.join(out_dir, 'frequency_vs_context_counts_with_residuals.pdf')
  plt.savefig(out_file, bbox_inches='tight')
  
if __name__ == '__main__':
  main()