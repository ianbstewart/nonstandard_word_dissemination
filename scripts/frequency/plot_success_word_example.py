"""
Plot example of successful innovation to contrast with 
a failure innovation.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style('white')
sns.set(style='ticks', context='paper')
from argparse import ArgumentParser
import pandas as pd
from scipy.stats import linregress
import os
from datetime import datetime

# success candy dates: afaik, boi, boyz, cringy, cuck, dank, defo, dgaf, dope, fave, fked, fwiw, ghosting, kinda, shitpost
# failure candy dates: adorbs, amazeballs, btw, dudes, dunno, fuckwit, fukken, lolno, omw, plz, selfie, shitlord, sorta, subbers, tyvm

def main():
  parser = ArgumentParser()
  parser.add_argument('--success_word', default='kinda')
  parser.add_argument('--tf_file', default='../../data/frequency/2013_2016_tf_norm_log.tsv')
  parser.add_argument('--out_dir', default='../output/')
  args = parser.parse_args()
  success_word = args.success_word
  tf_file = args.tf_file
  out_dir = args.out_dir
  
  ## load data
  tf = pd.read_csv(tf_file, sep='\t', index_col=0)
#   date_fmt = '%Y-%m'
#   tf.columns = map(lambda x: datetime.strptime(x, date_fmt), tf.columns)
  success_tf = tf.loc[success_word, :]
  
  ## compute best-fit line
  X = pd.np.arange(len(success_tf.index))
  Y = success_tf.values
  m, b, r, p, err = linregress(X, Y)
  Y_fit = m*X + b
  
  ## plot
  success_color = 'r'
  best_fit_color = 'b'
  success_linestyle = '-'
  best_fit_linestyle = '--'
  width = 5.5
  height = 3.0
  label_size = 18
  tick_size = 12
  title_size = 24
  x_label = 'Date'
  y_label = 'log(f)'
  x_tick_interval = 9
  x_ticks, x_tick_labels = zip(*zip(X, tf.columns)[::x_tick_interval])
  plt.figure(figsize=(width, height))
  plt.plot(X, Y, color=success_color, linestyle=success_linestyle)
  plt.plot(X, Y_fit, color=best_fit_color, linestyle=best_fit_linestyle)
  plt.xlabel(x_label, size=label_size)
  plt.ylabel(y_label, size=label_size)
  plt.xticks(x_ticks, x_tick_labels, fontsize=tick_size)
  plt.yticks(fontsize=tick_size)
  plt.title(success_word, fontsize=title_size)
  plt.tight_layout()
#   sns.regplot(success_tf.index, success_tf, fit_reg=True, ci=None, line_kws={'color':success_line_color})
  
  ## save
  out_file = os.path.join(out_dir, '%s_success_word_example.png'%(success_word))
  plt.savefig(out_file, bbox_inches='tight')

if __name__ == '__main__':
  main()