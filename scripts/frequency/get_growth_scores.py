"""
Compute growth scores for all words 
using Spearman's correlation coefficient.
"""
import pandas as pd
from scipy.stats import spearmanr
from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--tf_file', default='../../data/frequency/2013_2016_tf_norm_log.tsv')
    parser.add_argument('--out_dir', default='../../data/frequency/')
    args = parser.parse_args()
    tf_file = args.tf_file
    out_dir = args.out_dir
    
    ## load data
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    tf.fillna(tf.min().min(), inplace=True)
    
    ## compute growth
    N = tf.shape[1]
    X = pd.np.arange(N)
    growth_params = tf.apply(lambda y: pd.Series(spearmanr(X, y)), axis=1)
    growth_params.columns = ['spearman', 'pval']
    
    ## write to file
    out_file = os.path.join(out_dir, 'growth_scores.tsv')
    growth_params.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()