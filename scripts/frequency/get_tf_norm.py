"""
Convert raw frequencies to normalized
probabilities for easier comparison.
"""
import pandas as pd
import argparse
import os
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_default_vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_tf', 
                        default='../../data/frequency/2013_2016_tf.tsv')
    parser.add_argument('--out_file', default=None)
    parser.add_argument('--vocab', default='ALL')
    args = parser.parse_args()
    raw_tf_file = args.raw_tf
    raw_tf = pd.read_csv(raw_tf_file, sep='\t', index_col=0)
    vocab = args.vocab
    if(vocab != 'ALL'):
        vocab = get_default_vocab()
        raw_tf = raw_tf.loc[vocab]
    sums = raw_tf.sum(axis=0)
    norm_tf = raw_tf / sums
    # smooth and log
    norm_tf += norm_tf[norm_tf > 0].min().min()
    log_tf = pd.np.log10(norm_tf)
    if(args.out_file is None):
        out_dir = os.path.dirname(raw_tf_file)
        new_name = os.path.basename(raw_tf_file).replace('tf', 'tf_norm_log')
        out_file = os.path.join(out_dir, new_name)
        log_tf.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
