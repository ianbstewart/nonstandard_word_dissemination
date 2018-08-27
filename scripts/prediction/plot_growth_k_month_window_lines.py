"""
In which we plot the accuracy
results from the binary success
prediction task, using lines for 
each model: x axis = months of training.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd
import os
from itertools import cycle

def main():
    parser = ArgumentParser()
    parser.add_argument('--accuracy_result_file', default='../../output/success_1_12_window.tsv')
    parser.add_argument('--out_dir', default='../../output')
    args = parser.parse_args()
    accuracy_result_file = args.accuracy_result_file
    out_dir = args.out_dir
    
    ## load data
    accuracy_results = pd.read_csv(accuracy_result_file, sep='\t', index_col=False)
    feat_names = accuracy_results.loc[:, 'feat_names'].unique()
    k_vals = sorted(accuracy_results.loc[:, 'k'].unique())
    # assume that first k value actually represents 1 month of data
    k_vals = map(lambda x: x-min(k_vals)+1, k_vals)
    
    ## get mean, sd
    acc_metric = 'accuracy'
    accuracy_results.loc[:, acc_metric] *= 100
    accuracy_means = accuracy_results.groupby(['feat_names', 'k']).apply(lambda x: x.loc[:, acc_metric].mean())
    accuracy_sd = accuracy_results.groupby(['feat_names', 'k']).apply(lambda x: x.loc[:, acc_metric].std() / x.shape[0]**.5)
    
    ## plot
    bar_col = 'b'
    err_col = 'k'
    x_label = 'Months of training'
    y_label = acc_metric.capitalize()
    label_size = 14
    line_colors = cycle(['b', 'r', 'k', 'g'])
    line_styles = cycle(['-', '--', '-.', ':'])
    height = 2.5
    width = 4
    plt.figure(figsize=(width, height))
    for feat_name in feat_names:
        line_color = next(line_colors)
        line_style = next(line_styles)
        accuracy_means_f = accuracy_means.loc[feat_name]
        accuracy_sd_f = accuracy_sd.loc[feat_name]
        plt.plot(k_vals, accuracy_means_f, color=line_color, linestyle=line_style, label=feat_name)
        # error bars
        plt.errorbar(k_vals, accuracy_means_f, yerr=accuracy_sd_f, color=line_color, capsize=1, fmt='', linestyle=line_style)
    plt.xlabel(x_label, fontsize=label_size)
    plt.ylabel(y_label, fontsize=label_size)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    ## save
    out_file_name = os.path.basename(accuracy_result_file).replace('.tsv', '_lines.pdf')
    out_file = os.path.join(out_dir, out_file_name)
    plt.savefig(out_file)
    
if __name__ == '__main__':
    main()
