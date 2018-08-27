"""
Compute social variable diffusion, following
the formula from Altmann et al. (2011):
diffusion(word) = (unique # social vals that used word at least once) / sum(1 - e^(freq(word)*(number of words contributed by user)))
"""
from __future__ import division
import pandas as pd
import argparse
import os
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_default_vocab
import math
from collections import defaultdict, Counter

def diffusion_exact(vocab_social_counts, social_word_counts, vocab_tf):
    """
    Compute exact (by Altmann et al. 2011) diffusion of 
    all provided vocab words.
    
    Parameters:
    -----------
    vocab_social_counts : pandas.Series
    Unique number of social vals per vocab word.
    social_word_counts : pandas.Series
    Number of words per social val.
    vocab_tf : pandas.Series
    Normalized frequency per vocab word.
    
    Returns:
    --------
    diffusion : pandas.Series
    Diffusion per vocab word.
    """
    denom = (1 - pd.np.exp(pd.np.outer(social_word_counts, vocab_tf))).sum(axis=0)
    diffusion = vocab_social_counts / denom
    return diffusion

def diffusion_approx(vocab_social_counts, social_word_counts, vocab_tf):
    """
    Approximate diffusion with McLaurin expansion.
    
    Parameters:
    -----------
    vocab_social_counts : pandas.Series
    Unique number of social vals per vocab word.
    social_word_counts : pandas.Series
    Number of words per social val.
    vocab_tf : pandas.Series
    Normalized frequency per vocab word.
    
    Returns:
    --------
    diffusion : pandas.Series
    Diffusion per vocab word.
    """
    m1 = social_word_counts.sum()
    m2 = (social_word_counts**2).sum()
    m3 = (social_word_counts**3).sum()
    denom = vocab_tf * m1 - .5 * (vocab_tf ** 2) * m2 + (1/6)*(vocab_tf ** 3) * m3
    diffusion = vocab_social_counts / denom
    # replace inf and nan with zeros because due to vocab_tf=0
    nans = pd.np.isnan(diffusion)
    nans = nans[nans].index
    diffusion[nans] = 0
    diffusion.replace(pd.np.inf, 0, inplace=True)
    return diffusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/frequency/')
    parser.add_argument('--social_var', 
                        # default='user')
                        default='subreddit')
                        # default='thread')
    parser.add_argument('--tf', default='../../data/frequency/2015_2016_tf_norm.tsv')
    parser.add_argument('--all_dates', nargs='+', default=None)
    args = parser.parse_args()
    data_dir = args.data_dir
    social_var = args.social_var
    tf_file = args.tf
    all_dates = args.all_dates
    vocab = get_default_vocab()
    social_var_count_file = os.path.join(data_dir, '2015_2016_%s_unique.tsv'%(social_var))
    social_var_counts = pd.read_csv(social_var_count_file, sep='\t', index_col=0)
    vocab = list(set(vocab) & set(social_var_counts.index.tolist()))
    print('got %d final vocab'%(len(vocab)))
    social_word_count_file = os.path.join(data_dir, '2015_2016_%s_words.tsv'%(social_var))
    social_word_counts = pd.read_csv(social_word_count_file, sep='\t', index_col=0)
    if(all_dates is None):
        all_dates = sorted(social_word_counts.columns)
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    all_social_vals = social_word_counts.index.tolist()
    # all_diffusion_vals = defaultdict(Counter)
    cutoff = 200
    for d in all_dates:
        #print('relevant social var counts %s'%
        #       (social_var_counts[d]))
        # all_sums = (1 - math.e ** (tf[d] * social_var_counts[d]))
        relevant_social_var_counts = social_var_counts[d]
        relevant_social_word_counts = social_word_counts[d]
        all_diffusion_vals = defaultdict(Counter)
        # vectorizing!
        vocab_tf = tf[d]
        vocab_social_counts = relevant_social_var_counts.loc[vocab]
        # diffusion = diffusion_exact(vocab_social_counts, relevant_social_word_counts, vocab_tf)
        diffusion = diffusion_approx(vocab_social_counts, relevant_social_word_counts, vocab_tf)
        diffusion = pd.DataFrame(diffusion, index=vocab, columns=[d])
        print('got diffusion %s'%(diffusion))
        # replace inf and NaN values?
        
        # unvectorized
        # for i, v in enumerate(vocab):
        #     v_tf = tf[d].loc[v]
        #     if(v_tf > 0):
        #         v_social_count = relevant_social_var_counts.loc[v]
        #         # compute social val sum
        #         denom = (1 - math.e**(-v_tf * relevant_social_word_counts)).sum()
        #         diffusion = v_social_count / denom
        #     else:
        #         diffusion = 0
        #     all_diffusion_vals[d][v] = diffusion
        #     if(i % 100 == 0):
        #         print('processed %d vocab'%(i))
            # if(i >= cutoff):
            #     break
        # write to file!
        out_fname = os.path.join(data_dir, '%s_%s_diffusion.tsv'%(d, social_var))
        diffusion.to_csv(out_fname, sep='\t')
        # all_diffusion_vals = pd.DataFrame(all_diffusion_vals)
        # all_diffusion_vals.to_csv(out_fname, sep='\t')
    # write to combined file
    # out_fname = os.path.join(data_dir, '2015_2016_%s_diffusion.tsv'%(social_var))
    # all_diffusion_vals = pd.DataFrame(all_diffusion_vals)
    # all_diffusion_vals.to_csv(out_fname, sep='\t')
    
if __name__ == '__main__':
    main()
