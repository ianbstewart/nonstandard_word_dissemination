"""
Approximate context diversity for each word w
by collecting all ngrams containing w,
repeatedly sampling ngrams weighted by frequency, 
then counting unique ngrams in sample.
"""
from __future__ import division
from multiprocessing import Pool
from argparse import ArgumentParser
from time import time
import pandas as pd
from itertools import izip, repeat
from math import ceil

def get_context_counts(w, count_file):
    """
    Get context counts for contexts that 
    contain word w.
    
    Parameters:
    -----------
    w : str
    count_file : str
    
    Returns:
    --------
    context_counts : {str : int}
    """
    context_counts = {}
    for i, l in enumerate(open(count_file, 'r')):
        # skip first line
        if(i > 0):
            l, count = l.strip().split('\t')
            l_txt = l.split(' ')
            if(w in l_txt):
                count = int(count)
                context_counts[l] = count
            if(i % 10000000 == 0):
                print('processed %d lines for word %s'%(i, w))
    return context_counts

def get_context_counts_default(args):
    return get_context_counts(*args)

def main():
    parser = ArgumentParser()
    parser.add_argument('word_file')
    parser.add_argument('ngram_file')
    parser.add_argument('out_file')
    # sample size
    parser.add_argument('--S', type=int, default=50)
    # number of sample iters
    parser.add_argument('--I', type=int, default=100)
    # number of processes to run at same time
    parser.add_argument('--P', type=int, default=100)
    args = parser.parse_args()
    word_file = args.word_file
    ngram_file = args.ngram_file
    out_file = args.out_file
    S = args.S
    I = args.I
    P = args.P
    words = pd.read_csv(word_file, index_col=None)['word'].tolist()
    N = len(words)
    # set up processor and data
    pool = Pool(processes=P)
    chunk_size = int(ceil(N / P))
    # time how long it takes to collect context counts
    start = time()
    sample_zipped = izip(words, repeat(ngram_file))
    context_count_results = pool.map(get_context_counts_default, sample_zipped, chunksize=chunk_size)
    pool.close()
    end = time()
    elapsed = end - start
    print('time elapsed: %d words = %.3f seconds'%(N, elapsed))
    # now compute sample context counts!
    context_samples_unique_pcts = {}
    for i, (w, context_counts) in enumerate(izip(words, context_count_results)):
        context_counts = pd.Series(context_counts)
        context_probs = pd.np.array(context_counts)
        context_probs = context_probs / context_probs.sum()
        context_list = context_counts.index.tolist()
        sample_size = min(S, len(context_list) / 2)
        context_samples = [pd.np.random.choice(context_list, S, replace=True, p=context_probs) for i in range(I)]
        context_samples_unique = pd.np.array(map(lambda x: len(set(x)) / len(x), context_samples))
        context_samples_unique_pcts[w] = context_samples_unique
    context_samples_vals = {'word' : [], 'CC_mean' : [], 'CC_err' : []}
    for w, context_samples_unique in sorted(context_samples_unique_pcts.items(), key=lambda x: pd.np.mean(x[1])):
        context_count_mean = pd.np.mean(context_samples_unique)
        context_count_stderr = pd.np.std(context_samples_unique) / I ** 0.5
        context_samples_vals['word'].append(w)
        context_samples_vals['CC_mean'].append(context_count_mean)
        context_samples_vals['CC_err'].append(context_count_stderr)
        # print('%s = %.3f +/- %.3f'%(w, context_count_mean, context_count_stderr))
    context_samples_df = pd.DataFrame(context_samples_vals)
    context_samples_df.index = context_samples_df['word']
    context_samples_df.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
