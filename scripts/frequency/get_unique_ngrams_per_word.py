"""
Count unique number of n-gram
contexts for each word in vocab.
"""
from __future__ import division
import pandas as pd
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_default_vocab
import argparse
import os, re
from collections import Counter
from zipfile import ZipFile
from stopwords import get_stopwords

PUNCT=re.compile('[\.!?//\\\\]')
def get_unique_ngram_counts(line_iter, n, ngram_pos=1):
    """
    Get unique ngram counts for every 
    unigram word in ngram vocabulary.
    Convert to percents to compare
    counts across dates.
    
    Parameters:
    -----------
    line_iter : [str] or generator
    Generate each line in ngram file.
    n : int
    ngram_pos : int
    Position within each ngram to count as the "center" word.
    E.g. if ngram_pos = 1, then the trigram "i ate the" will
    have "ate" as the "center" word to be counted.
    
    Returns:
    --------
    counts : pandas.DataFrame
    Rows = words, columns = dates (should be one date).
    unique_ngrams : int
    Number of unique ngrams in full file.
    """
    counts = Counter()
    # cutoff = 1000000
    for i, l in enumerate(line_iter):
        # assume that file starts with date
        if(i == 0):
            timeframe = l.strip()
        elif(l.strip() != ''):
            try:
                ngram, count = l.split('\t')
                # check for punctuation
                if(not PUNCT.search(ngram)):
                    words = ngram.split(' ')
                    # special case: 
                    # restrict gram counts to
                    # only the specified position
                    # if(n > 2):
                    words = [words[ngram_pos]]
                    for w in words:
                        counts[w] += 1
                # count = int(count)
                # counts[ngram] = count
            except Exception, e:
                print('stopped counting ngrams because %s'%(e))
                break
        if(i % 1000000 == 0):
            print('processed %d ngrams'%(i))
    unique_ngrams = i
    counts = pd.DataFrame(counts, index=[timeframe]).transpose()
    # convert to percents!
    # counts /= unique_ngrams
    print('date %s has counts\n%s'%(timeframe, counts.head()))
    return counts, unique_ngrams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ngram_files', nargs='+')
    parser.add_argument('--n', type=int,
                        # default=3)
                        default=2)
    parser.add_argument('--data_dir', default='../../data/frequency/')
    parser.add_argument('--timeframe', default='2015_2016')
    # parser.add_argument('--tf_file', default='../../data/frequency/2013_2016_tf.tsv')
    parser.add_argument('--ngram_pos', type=int, default=1)
    parser.add_argument('--file_suffix', default=None)
    args = parser.parse_args()
    ngram_files = args.ngram_files
    n = args.n
    data_dir = args.data_dir
    combined_timeframe = args.timeframe
    # tf_file = args.tf_file
    ngram_pos = args.ngram_pos
    file_suffix = args.file_suffix
    # collect all n-gram files
    # file_suffix = '%dgram_tf.tsv'%(n)
    # ngram_tf_files = [os.path.join(data_dir, f) 
     #                  for f in os.listdir(data_dir)
      #                if re.findall(file_suffix, f)]
    vocab = get_default_vocab()
    # add stopwords
    stops = get_stopwords('en')
    vocab = list(set(vocab) | set(stops))
    ngram_files = sorted(ngram_files)
    print('got all ngram files %s'%(str(ngram_files)))
    # load tf
    # tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    # tf = tf.loc[vocab]
    # unzip one file at a time and collect counts
    all_counts = []
    all_pcts = []
    for ngram_file in ngram_files:
        timeframe = re.findall('201[0-9]-[0-9]{2}', ngram_file)[0]
        with open(ngram_file, 'r') as line_iter:
            print(ngram_file)
            counts, unique_ngrams = get_unique_ngram_counts(line_iter, n, ngram_pos=ngram_pos)
            # restrict vocab
            count_vocab = list(set(vocab) & set(counts.index))
            counts = counts.loc[count_vocab, :]
            # cleanup: ensure that context count is always lower than tf
            # update: don't do this because it's LYING
            # count_tf_combined = pd.concat([counts.loc[:, [timeframe]], tf.loc[count_vocab, [timeframe]]], axis=1)
            # count_tf_combined.columns = ['ngram', 'f']
            # count_tf_combined['ngram'] = count_tf_combined.min(axis=1)
            # counts[timeframe] = count_tf_combined['ngram']
            all_counts.append(counts)
            pcts = counts / unique_ngrams
            all_pcts.append(pcts)
    unique_ngram_tf = pd.concat(all_counts, axis=1)
    print('got unique counts\n%s'%(unique_ngram_tf.head()))
    unique_ngram_pcts = pd.concat(all_pcts, axis=1)
    # include timeframe, n, ngram position and optional suffix in filename
    suffix = '.tsv'
    if(file_suffix is None):
        suffix = '%s.tsv'%(file_suffix)
    fbase = '%s_unique_%dgram_%dpos_counts%s'%(combined_timeframe, n, ngram_pos, suffix)
    # write to file
    count_fname = os.path.join(data_dir, fbase)
    unique_ngram_tf.to_csv(count_fname, sep='\t')
    pct_fname = os.path.join(data_dir, '%s_unique_%dgram_pcts%s'%(combined_timeframe, n, suffix))
    unique_ngram_pcts.to_csv(pct_fname, sep='\t')

if __name__ == '__main__':
    main()
