"""
Compute the basic stats over the provided corpus:
- comments
- users
- tokens
- subreddits
- users
- threads
"""
from bz2 import BZ2File
from collections import defaultdict
import pandas as pd
from itertools import izip
from argparse import ArgumentParser
import re, os

def get_corpus_stats(corpus_file, corpus_meta_file):
    """
    Compute comment, token, user, subreddit and thread
    counts over the corpus file. Assumes that corpus and
    meta file are aligned!
    
    Parameters:
    -----------
    corpus_file : str
    corpus_meta_file : str
    
    Returns:
    --------
    stats : pandas.Series
    unique_stats : {str : set}
    """
    stats = defaultdict(float)
    unique_stats = defaultdict(set)
    # cutoff = 100000
    with BZ2File(corpus_file, 'r') as corpus, BZ2File(corpus_meta_file, 'r') as corpus_meta:
        for i, (l, m) in enumerate(izip(corpus, corpus_meta)):
            txt = l.split(' ')
            _, user, thread, sub, _, _, _ = m.split('\t')
            stats['comments'] += 1
            stats['tokens'] += len(txt)
            unique_stats['user'].add(user)
            unique_stats['thread'].add(thread)
            unique_stats['subreddit'].add(sub)
            if(i % 1000000 == 0):
                print('%d comments processed'%(i))
            # if(i >= cutoff):
              #   break
    unique_stat_counts = {k : len(v) for k,v in unique_stats.iteritems()}
    stats.update(unique_stat_counts)
    stats = pd.Series(stats)
    return stats, unique_stats

def main():
    parser = ArgumentParser()
    parser.add_argument('corpora', nargs='+')
    parser.add_argument('--out_dir', default='../../data/metadata/')
    args = parser.parse_args()
    corpus_files = args.corpora
    out_dir = args.out_dir
    # first get individual stats
    joint_stats = defaultdict(float)
    joint_unique_stats = defaultdict(set)
    for corpus_file in corpus_files:
        corpus_meta_file = corpus_file.replace('.bz2', '_meta.bz2')
        stats, unique_stats = get_corpus_stats(corpus_file, corpus_meta_file)
        joint_stats = {k : joint_stats[k] + v for k,v in stats.iteritems()}
        joint_unique_stats = {k : joint_unique_stats[k] | v for k,v in unique_stats.iteritems()}
        timeframe = re.findall('201[0-9]-[0-9]{2}', corpus_file)[0]
        out_file = os.path.join(out_dir, '%s_corpus_stats.tsv'%(timeframe))
        stats = pd.DataFrame(stats, columns=[timeframe])
        stats.to_csv(out_file, sep='\t')
    # then get joint stats
    joint_unique_stats = {k : len(v) for k,v in joint_unique_stats.iteritems()}
    joint_stats.update(joint_unique_stats)
    joint_stats = pd.Series(joint_stats)
    file_dates = [re.findall('201[0-9]', f)[0] for f in sorted(corpus_files)]
    timeframe = file_dates[0] + '_' + file_dates[-1]
    print('full timeframe %s'%(timeframe))
    out_file = os.path.join(out_dir, '%s_corpus_stats.tsv'%(timeframe))
    joint_stats.to_csv(out_file, sep='\t')
    
if __name__ == '__main__':
    main()
