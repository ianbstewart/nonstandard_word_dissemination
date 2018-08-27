"""
Get word counts for social categories in social-term-matrix
which we can later use for PMI calculations.
"""
import pandas as pd
import numpy as np
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_all_comment_files, get_default_stopwords, get_default_tokenizer, get_default_vocab
from nltk.tokenize.regexp import WhitespaceTokenizer
from bz2 import BZ2File
import argparse
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
import re, os
from itertools import izip

def write_full_social_dtm(dtm, all_social_vals, vocab, 
                          date_str, social_var, out_dir):
    """
    Write full social DTM to file with separate 
    column (word) and row (social vals) files.
    Parameters:
    -----------
    dtm : scipy.sparse.csr.csr_matrix
    Sparse DTM, rows = social var, cols = vocab.
    all_social_vals : [str]
    All social values, e.g. all usernames (in DTM order).
    vocab : [str]
    All vocabulary types (in DTM order).
    date_str : str
    social_var : str
    out_dir : str
    """
    out_fname = os.path.join(out_dir, '%s_%s_dtm'%(date_str, social_var))
    print('writing to file %s'%(out_file))
    np.savez(out_fname, data=dtm, indices=dtm.indices, 
             indptr=dtm.indptr, shape=dtm.shape)
    # write users and vocab separately
    col_fname = os.path.join(out_dir, '%s_%s_dtm.cols'%(date_str, social_var))
    with open(col_fname, 'w') as col_file:
        for v in vocab:
            col_file.write('%s\n'%(v))
    row_fname = os.path.join(out_dir, '%s_%s_dtm.rows'%(date_str, social_var))
    with open(row_fname, 'w') as row_file:
        for u in all_users:
            row_file.write('%s\n'%(u))

def get_social_word_counts(social_var, 
                           vocab, comment_file, 
                           meta_file, comment_thresh=10):
    """
    Compute unique number of social vars 
    per word in vocab over all comments.
    Parameters:
    -----------
    social_var : str
    vocab : [str]
    Vocabulary to count.
    comment_file : str
    meta_file : str
    Tab-separated metadata file containing comment date, 
    author, thread ID, and subreddit.
    comment_thresh : int
    Minimum number of comments for a social var to be counted.
    Returns:
    --------
    social_var_counts : numpy.array
    """
    # indices in meta file corresponding to social vars
    social_var_indices = {'user' : 1, 'subreddit' : 3, 'thread' : 2}
    social_txt = defaultdict(list)
    tokenizer = WhitespaceTokenizer()
    stopwords = get_default_stopwords()
    ngram_range = (1,1)
    min_df = 1
    cv = CountVectorizer(encoding='utf-8', lowercase=True, tokenizer=tokenizer.tokenize,
                         stop_words=stopwords, ngram_range=ngram_range, 
                         min_df=min_df, 
                         vocabulary=vocab,
                         binary=True)
    # keep it simple and store {vocab : {sub : count}}
    social_word_counts = defaultdict(Counter)
    with BZ2File(comment_file, 'r') as comments, BZ2File(meta_file, 'r') as metas:
        for i, (comment, meta) in enumerate(izip(comments, metas)):
            meta = meta.split('\t')
            social_id = meta[social_var_indices[social_var]]
            # print('got social id %s'%(social_id))
            # social_txt[social_id].append(comment)
            for w in tokenizer.tokenize(comment):
                social_word_counts[w][social_id] += 1
            if(i % 100000 == 0):
                print('processed %d comments'%(i))
            # if(i == 500000):
            #     break
    social_word_counts = {w : d for w,d in social_word_counts.iteritems() if w in vocab}
    social_word_counts = {w : {k : v for k,v in d.iteritems() if v >= comment_thresh} for w,d in social_word_counts.iteritems()}
    social_word_counts = {w : len(d) for w,d in social_word_counts.iteritems()}
    social_word_counts = np.array([social_word_counts[v] if v in social_word_counts else 0. for v in vocab])

    # old code for constructing word/social dtm
    # restrict to consistent users??
    # social_txt = {k : v for k,v in social_txt.items() 
    #               if len(v) >= comment_thresh}
    # # now convert to DTM
    # def get_txt_iter(social_txt):
    #     N = len(social_txt)
    #     for i, v in enumerate(social_txt.itervalues()):
    #         if(i % 1000 == 0):
    #             print('processed %d/%d social vars'%(i, N))
    #         yield ' '.join(v)
    # txt_iter = get_txt_iter(social_txt) 
    # # txt_iter = (' '.join(v) for v in social_txt.values())
    # dtm = cv.fit_transform(txt_iter)
    # print('got %s dtm %s'%(social_var, dtm))
    # # save sparse matrix
    # # all_social_vals = social_txt.keys()
    # # vocab = sorted(cv.vocabulary_, key=lambda x: cv.vocabulary_[x])
    # # comment_date = re.findall(r'201[0-9]-[0-9]+', comment_file)[0]
    # # write_full_social_dtm(dtm, all_social_vals, vocab, comment_date, social_var)
    # # save unique social count for each word
    # # combine all counts per word
    # social_word_counts = np.array(dtm.sum(axis=0)).flatten()
    return social_word_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_files', nargs='+', default=None)
    parser.add_argument('--out_dir', default='../../data/frequency/')
    parser.add_argument('--social_vars', nargs='+', 
                       # default=['user', 'thread', 'subreddit'])
                        # default=['user'])
                        # default=['thread'])
                        default=['subreddit'])
    args = parser.parse_args()
    comment_files = args.comment_files
    out_dir = args.out_dir
    social_vars = args.social_vars
    if(comment_files is None):
        # data_dir = '/mnt/new_hg190/corpora/reddit_comment_data/monthly_submission/'
        data_dir = '/mnt/new_hg190/corpora/reddit_comment_data/monthly_submission/'
        years = ['2015', '2016']
        comment_files = get_all_comment_files(data_dir, years)
        print('comment files %s'%(str(comment_files)))
        # but we actually want clean_normalized lol
        comment_files = [f.replace('.bz2', '_normalized.bz2') 
                         for f in comment_files]
    meta_files = [f.replace('.bz2', '_meta.bz2') for f in comment_files]
    # print('got meta files %s'%(meta_files))
    # TODO: start small, eventually move to rest of files
    # comment_files = comment_files[3:]
    # comment_files = comment_files[1:]
    
    # for testing
    # social_vars = social_vars[:1]

    vocab = get_default_vocab()
    # chunk_size = 1000
    # chunk_size = 5000
    # chunk_size = len(vocab)
    # chunks = int(len(vocab) / chunk_size)
    # vocab_chunks = [vocab[i*chunk_size:i*chunk_size+chunk_size] 
    #                 for i in xrange(chunks)]
    # start small
    # top_vocab = 1000
    top_vocab = 100000
    stopwords = get_default_stopwords()
    # already whitespace separated, so just need whitespace tokenizer
    tokenizer = WhitespaceTokenizer()
    ngram_range = (1,1)
    min_df = 1
    cv = CountVectorizer(encoding='utf-8', lowercase=True, tokenizer=tokenizer.tokenize,
                         stop_words=stopwords, ngram_range=ngram_range, 
                         min_df=min_df, 
                         # max_features=top_vocab,
                         vocabulary=vocab,
                         # binarize to save space b/c we only care about cooccurrence
                         binary=True)
    out_dir = args.out_dir
    # min number of comments within social value
    # to make it count
    # social_comment_thresh = 10
    social_comment_thresh = 1
    for comment_file, meta_file in izip(comment_files, meta_files):
        print('processing comment file %s and meta file %s'%
              (comment_file, meta_file))
        date_str = re.findall(r'201[0-9]-[0-9]+', comment_file)[0]
        for social_var in social_vars:
            # use for full dtm
            # out_fname = os.path.join(out_dir, '%s_%s_dtm'%(date_str, social_var))
            out_fname = os.path.join(out_dir, '%s_%s_unique.tsv'%(date_str, social_var))
            # for each vocab chunk in list, get unique social counts!
            # for vocab in vocab_chunks:
            print('got vocab size %d'%(len(vocab)))
            social_word_counts = get_social_word_counts(social_var, 
                                                        vocab, comment_file, 
                                                        meta_file, 
                                                        comment_thresh=social_comment_thresh)
            # write to file
            social_word_counts = pd.DataFrame(social_word_counts,
                                              index=vocab)
            social_word_counts.to_csv(out_fname, sep='\t', header=False)
            # append to combined file
            # with open(out_fname, 'a') as out_file:
            #     for v, c in zip(vocab, social_word_counts):
            #         print('got count %s'%(str(c)))
            #         out_file.write('%s\t%d\n'%(v, c))

if __name__ == '__main__':
    main()