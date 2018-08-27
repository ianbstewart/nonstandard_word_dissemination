"""
Get cooccurrence between unigram vocabulary and
ngram contexts. 
"""
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_default_vocab, get_all_comment_files
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.regexp import WhitespaceTokenizer
import argparse
import os, re
from bz2 import BZ2File
import pandas as pd
from collections import Counter
import random

def get_ngram_counts(comment_iter, n, tokenizer=None, sample_pct=100):
    """
    Compute ngram counts from comments.
    
    Parameters:
    -----------
    comment_iter : generator
    n : int
    tokenizer : nltk.tokenize.Tokenizer
    sample_pct : float
    Optional percentage from which to subsample the data.
    
    Returns:
    --------
    counts : pandas.DataFrame
    Rows = ngrams, col = counts.
    """
    if(tokenizer is None):
        tokenizer = WhitespaceTokenizer()
    counts = Counter()
    for i, c in enumerate(comment_iter):
        if(sample_pct == 100 or random.random()*100 < sample_pct):
            ngrams = ngram_split(c, n, tokenizer)
            for ngram in ngrams:
                ngram = [' '.join(ngram)]
                counts.update(ngram)
        if(i % 1000000 == 0):
            print('got %d unique ngrams'%(len(counts)))
    # convert to dataframe
    counts = pd.DataFrame(pd.Series(counts))
    return counts

def make_iter(txt_iter):
    """
    Make iterator over text
    documents and provide regular
    count output.
    """
    ctr = 0
    for c in txt_iter:
        if(ctr % 1000000 == 0):
            print('%d comments processed'%(ctr))
        ctr += 1
        yield c

def ngram_split(txt, n, tokenizer):
    txt_split = tokenizer.tokenize(txt)
    # add start/end tokens
    txt_split = ['<START>'] + txt_split + ['<END>']
    ngrams = [txt_split[i:i+n] for i in range(len(txt_split)-n+1)]
    return ngrams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='../../data/frequency')
    parser.add_argument('--comment_files', nargs='+', default=None)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--file_suffix', default=None)
    parser.add_argument('--sample_pct', type=float, default=100)
    args = parser.parse_args()
    out_dir = args.out_dir
    comment_files = args.comment_files
    n = args.n
    file_suffix = args.file_suffix
    sample_pct = args.sample_pct
    if(comment_files is None):
        comment_files = get_all_comment_files()
        # replace with clean normalized (smaller vocab)
        comment_files = [f.replace('.bz2', '_clean_normalized.bz2') 
                         for f in comment_files]
    # start small
    # comment_files = comment_files[:1]
    # min_df = 5
    # min_tf = 10
    min_tf = 1
    stopwords = []
    tokenizer = WhitespaceTokenizer()
    # breaking memory
    # ngram_range = (1,3)
    # ngram_range = (2,3)
    # ngram_range = (2,2)
    # ngram_range = (1,1)
    # no CountVectorizer because memory and we don't need
    # cooccurrence anyway
    # cv = CountVectorizer(min_df=min_df, tokenizer=tokenizer.tokenize,
    #                      stop_words=stopwords, ngram_range=ngram_range)
    date_format = '201[0-9]-[0-9]+'
    for f in comment_files:
        print('processing file %s'%(f))
        date_str = re.findall(date_format, f)[0]
        # for each level of ngram, recompute counts
        # for n in range(ngram_range[0], ngram_range[1]+1):
        print('computing ngram = %d'%(n))
        with BZ2File(f, 'r') as comment_file:
            # takes too long to generate full DTM...what do??
            # just compute counts
            comment_iter = make_iter(comment_file)
            counts = get_ngram_counts(comment_iter, n, tokenizer=tokenizer, sample_pct=sample_pct)
            
            # limit min_frequency?
            counts = counts[counts >= min_tf]
            counts.columns = [date_str]
            # write to file
            # TOO MUCH SPACE => compress?
            if(file_suffix is not None):
                out_fname = os.path.join(out_dir, '%s_%dgram_tf_%s.tsv'%(date_str, n, file_suffix))
            else:
                out_fname = os.path.join(out_dir, '%s_%dgram_tf.tsv'%(date_str, n))
            counts.to_csv(out_fname, sep='\t')
            
if __name__ == '__main__':
    main()
