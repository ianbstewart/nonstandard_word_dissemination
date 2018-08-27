"""
Plot frequency vs. context counts and include
some example innovations.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from argparse import ArgumentParser
import os

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
  
  ## plot
  x_label = 'log(f)'
  y_label = 'log($C^{3}$)'
  label_size = 24
  tick_size = 18
  annotate_size = 14
  line_color = 'r'
  scatter_color = 'b'
  scatter_alpha = 0.1
  xy_offset = [-50, 30]
  arrow_width = 3
  arrow_color = 'k'
  plt.figure(figsize=(5, 5))
  sns.regplot(tf_sample, c3_sample, line_kws={'color': line_color}, scatter_kws={'color' : scatter_color, 'alpha' : scatter_alpha})
  plt.xlabel(x_label, fontsize=label_size)
  plt.ylabel(y_label, fontsize=label_size)
  plt.xticks(fontsize=tick_size)
  plt.yticks(fontsize=tick_size)
  for i in example_innovations:
    x = tf_mean.loc[i]
    y = c3_mean.loc[i]
    plt.annotate(i, xy=(x,y), xycoords='data', xytext=xy_offset, textcoords='offset points', fontsize=annotate_size,
                 arrowprops={'width':arrow_width, 'color':arrow_color})

  ## save
  out_file = os.path.join(out_dir, 'frequency_vs_context_counts_with_examples.pdf')
  plt.savefig(out_file, bbox_inches='tight')
  
if __name__ == '__main__':
  main()