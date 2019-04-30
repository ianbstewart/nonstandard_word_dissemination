"""
Compute and write to file the top-k vocab
to use in building embeddings. IMPORTANT so that
we can normalize corpus to have UNKs in all 
the right places!
"""
import pandas as pd
import os
import re
import argparse
import sys
from stopwords import get_stopwords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf',
                        # default='../../data/frequency/2015_2016_tf.tsv')
                        default='../../data/frequency/2015_2016_tf_norm.tsv')
    parser.add_argument('--top_k', type=int, default=100000)
    args = parser.parse_args()
    tf_file = args.tf
    top_k = args.top_k
    print(tf_file)
    timeframe = re.findall('201[0-9]_201[0-9]', tf_file)[0]
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    totals = tf.sum(axis=1)
    totals.sort_values(inplace=True, ascending=False)
    stops = set(get_stopwords('en'))
    # only want valid words!!
    valid_words = list(filter(lambda w: (type(w) is str and w.isalpha()) and w not in stops, totals.index))
    top_vocab = totals.loc[valid_words][:top_k]
    top_vocab = pd.DataFrame(top_vocab, columns=['count'])
    print('got %d vocab'%(len(top_vocab)))
    # renormalize
    top_vocab.loc[:, 'count'] = top_vocab.loc[:, 'count'] / top_vocab.loc[:, 'count'].sum(axis=0)
    out_dir = os.path.dirname(tf_file)
    out_fname = os.path.join(out_dir, '%s_top_%d_vocab.tsv'%(timeframe,top_k))
    top_vocab.to_csv(out_fname,sep='\t')
    
if __name__ == '__main__':
    main()
