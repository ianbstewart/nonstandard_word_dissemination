"""
Combine tf entries for words with duplicate 
characters (i.e. more than 3 of same 
character consecutively).
"""
import pandas as pd
import argparse
import re
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--tf', default='../../data/frequency/2015_2016_tf.tsv')
    parser.add_argument('--tf', default='../../data/frequency/2014_2016_tf.tsv')
    parser.add_argument('--max_len', type=int, default=3)
    args = parser.parse_args()
    tf_file = args.tf
    max_len = args.max_len
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    all_dates = sorted(tf.columns)
    tf = tf.ix[:,all_dates]
    vocab = tf.index.tolist()
    N = tf.shape[1]
    # new_tf = defaultdict(lambda : )
    new_tf = defaultdict(float)
    ctr = 0
    sub_str = r'([a-z])\1{%d,}'%(max_len)
    for i, v in enumerate(vocab):
        tf_v = tf.loc[v].values
        v = str(v)
        v_sub = re.sub(sub_str, r'\1\1\1', v)
        if(v_sub != v):
            # print('%s vs. %s'%(v_sub, v))
            ctr += 1
        new_tf[v_sub] += tf_v
        if(i % 10000 == 0):
            print('substituted %d/%d vocab'%(ctr, i))
    new_tf = pd.DataFrame(new_tf, index=all_dates).transpose()
    out_fname = tf_file.replace('tf', 'tf_sub')
    new_tf.to_csv(out_fname, sep='\t')

if __name__ == '__main__':
    main()
