"""
Plot concordance scores across feature sets.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os, re
# set TeX format
# but we can't! error when saving
# RuntimeError: dvipng was not able to process the following file:
# /nethome/istewart6/.cache/matplotlib/tex.cache/d6cc654e7e9c6b9a4029a7fa45df81b4.dvi
# plt.rcParams.update({'text.usetex' : True})

def main():
    parser = ArgumentParser()
    parser.add_argument('--covariate_score_file', default='../../output/cox_regression_concordance_10_fold_scores.tsv')
    parser.add_argument('--out_dir', default='../../output/')
    args = parser.parse_args()
    covariate_score_file = args.covariate_score_file
    out_dir = args.out_dir
    out_file = os.path.join(out_dir, 'survival_concordance_score_distribution.pdf')
    covariate_set_scores = pd.read_csv(covariate_score_file, sep='\t', index_col=0)
    covariate_set_names = covariate_set_scores.index.tolist()
    # sort covariate set names by length (i.e. f < f+C < f+S < f+C+S)
    covariate_set_names = sorted(covariate_set_names, key=lambda x: len(x))
    model_name_format = '$\mathtt{%s}$'
#     model_name_format = '\texttt{%s}'
    covariate_set_names_script = map(lambda n: model_name_format%(n), covariate_set_names)
    # collect and plot the scores
    score_names = [c for c in covariate_set_scores.columns if re.findall('score_[0-9]+', c)]
    plot_scores = [covariate_set_scores.loc[covariate_set_name, score_names] for covariate_set_name in covariate_set_names]
    label_size = 18
    tick_size = 12
    med_col = 'k'
    box_col = '0.5' # gray
    width = 3.
    height = 2.5
    dpi = 300 # resolution
    X_box = range(1, len(covariate_set_names)+1)
    plt.figure(figsize=(width, height))
    plt.boxplot(plot_scores, patch_artist=True, medianprops={'color' : med_col}, boxprops={'fc' : box_col})
    plt.xticks(X_box, covariate_set_names_script, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel('Feature set', fontsize=label_size)
    plt.ylabel('Concordance', fontsize=label_size)
    plt.tight_layout()
    # write to file
    plt.savefig(out_file, dpi=dpi)

if __name__ == '__main__':
    main()
