"""
Get frequency for tokens in all comments ion corpus.
"""
from sklearn.feature_extraction.text import CountVectorizer
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_all_comment_files, CommentIter, get_default_tokenizer, get_default_stopwords, MIN_COUNT, extract_year_month, get_non_english_communities, get_default_spammers, get_default_bots
from nltk.tokenize.casual import TweetTokenizer
from stopwords import get_stopwords
import pandas as pd
import numpy as np
import argparse
from collections import Counter
import os
from bz2 import BZ2File

def get_tf(comments):
    """
    Compute tf from comments, which
    we assume contain space-separated tokens.

    Parameters:
    -----------
    comments : generator

    Returns:
    --------
    tf : pandas.Series
    """
    tf = Counter()
    for c in comments:
        txt = c.strip().lower().split(' ')
        tf.update(txt)
        ctr += 1
        if(ctr % 1000000 == 0):
            print('%d comments processed'%(ctr))
    tf = pd.Series(tf)
    return tf
    
    return tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_files', nargs='+', default=None)
    parser.add_argument('--out_dir', default='../../data/frequency/')
    args = parser.parse_args()
    comment_files = args.comment_files
    out_dir = args.out_dir
    if(comment_files is None):
        data_dir = '/mnt/new_hg190/corpora/reddit_comment_data/monthly_submission/'
        years = ['2013', '2014', '2015', '2016']
        comment_files = get_all_comment_files(data_dir=data_dir,
                                              years=years)
        # assume that files are filtered and cleaned
        comment_files = [f.replace('.bz2', '_clean.bz2')]
    # don't need tokenizer, stopwords b/c it's faster
    # tokenizer = get_default_tokenizer()
    # stopwords = get_default_stopwords()
    # ngram_range = (1,1)
    # min_df = MIN_COUNT
    # cv = CountVectorizer(encoding='utf-8', lowercase=True, tokenizer=tokenizer.tokenize,
    #                      stop_words=stopwords, ngram_range=ngram_range, 
    #                      min_df=min_df)
    out_dir = '../../data/frequency/'
    ctr = 0
    
    for comment_file in comment_files:
        print('processing comment file %s'%(comment_file))
        year, month = extract_year_month(comment_file)
        comment_date = '%d-%02d'%(year, month)
        with BZ2File(comment_file, 'r') as comments:
            tf = get_tf(comments)
        print('collected %d token frequencies'%(len(tf)))
        tf = pd.DataFrame(tf, index=[comment_date]).transpose()
        # print('item sample %s'%(tf.items()[:10]))
        # filter for min count
        # tf.drop(tf[tf < min_df].index, 
        #                      inplace=True)
        # write to file!!
        out_file = os.path.join(out_dir, '%s_tf.tsv'%(comment_date))
        tf.to_csv(out_file, sep='\t', encoding='utf-8')

if __name__ == '__main__':
    main()
