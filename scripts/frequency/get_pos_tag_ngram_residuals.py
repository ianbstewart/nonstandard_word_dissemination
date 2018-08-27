"""
For each month of data: fit linear regression between 
raw frequency and unique POS ngram frequency, then
compute residuals.
"""
import pandas as pd
from scipy.stats import linregress
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_default_vocab, smooth_stats
from argparse import ArgumentParser
import os, re

def main():
    parser = ArgumentParser()
    parser.add_argument('tf_file')
    parser.add_argument('UP3_file')
    parser.add_argument('--out_dir', default='../../data/frequency/')
    args = parser.parse_args()
    tf_file = args.tf_file
    UP3_file = args.UP3_file
    out_dir = args.out_dir
    
    # load data
    vocab = get_default_vocab()
    print('got vocab')
    tf = pd.read_csv(tf_file, sep='\t', index_col=0) #, na_values=[' '], dtype=int)
    print('f loaded')
    tf = pd.np.log10(smooth_stats(tf.loc[vocab, :].fillna(0, inplace=False)))
    UP3 = pd.read_csv(UP3_file, sep='\t', index_col=0).loc[vocab].fillna(0, inplace=False)
    print('UP3 loaded')
    timeframe = re.findall('201[0-9]_201[0-9]', tf_file)[0]
    
    # fit regression for each month
    all_months = sorted(tf.columns)
    UP3_resids = []
    for d in all_months:
        tf_d = tf.loc[:, d]
        UP3_d = UP3.loc[:, d]
        N = tf_d.shape[0]
        print('bout to regress over data N=%d'%(N))
        m, b, r, p, err = linregress(tf_d, UP3_d)
        print('d=%s, UP3=%.3E*f + %.3E (R=%.3f, p=%.3E)'%(d, m, b, r, p))
        UP3_pred = m*tf_d + UP3_d
        UP3_resids_d = UP3_d - UP3_pred
        UP3_resids_d = pd.DataFrame(UP3_resids_d, columns=[d])
        print(UP3_resids_d.shape)
        UP3_resids.append(UP3_resids_d)
    
    # write to file
    UP3_resids = pd.concat(UP3_resids, axis=1)
    print(U3_resids.shape)
    out_file = os.path.join(out_dir, '%s_P3.tsv'%(timeframe))
    UP3_resids.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
