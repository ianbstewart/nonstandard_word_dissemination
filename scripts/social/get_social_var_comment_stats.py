"""
Compute total number of comments 
and words per social var.
WE NEED THIS TO APPROXIMATE DIFFUSION.
"""
import pandas as pd
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_all_comment_files
import argparse
from bz2 import BZ2File
from itertools import izip
import os, re
from collections import defaultdict

def get_social_var_comment_stats(social_var, comment_file, 
                                 meta_file, social_comment_thresh):
    """
    Get comment stats for given social var.
    
    Parameters:
    -----------
    social_var : str
    comments_file : str
    meta_file : str
    social_comment_thresh : int
    Minimum number of comments associated
    with a social val before social val can
    be counted.
    
    Returns:
    --------
    comment_stats : pandas.DataFrame
    Index = word, columns = stats.
    """
    social_var_indices = {'user' : 1, 'subreddit' : 3, 'thread' : 2}
    social_index = social_var_indices[social_var]
    comment_stats = defaultdict(lambda : {'words' : 0, 'comments' : 0})
    with BZ2File(comment_file, 'r') as comments, BZ2File(meta_file, 'r') as metas:
        for i, (comment, meta) in enumerate(izip(comments, metas)):
            meta = meta.split('\t')
            social_val = meta[social_index]
            words = len(comment.split(' '))
            comment_stats[social_val]['words'] += words
            comment_stats[social_val]['comments'] += 1
            if(i % 1000000 == 0):
                print('processed %d comments'%(i))
            # if(i > 5000000):
                # break
    # filter out the lesser social vals
    comment_stats = {s : counts for s, counts in comment_stats.items()
                     if counts['comments'] >= social_comment_thresh}
    print('got %d comment stats'%(len(comment_stats)))
    # convert to data frame
    social_vals, counts = zip(*comment_stats.items())
    word_counts = [c['words'] for c in counts]
    comment_counts = [c['comments'] for c in counts]
    comment_stats = pd.DataFrame({'words' : word_counts, 
                                  'comments' : comment_counts},
                                 index=social_vals)
    return comment_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_files', nargs='+', 
                        default=None)
    parser.add_argument('--social_vars', nargs='+',
                        # default=['user', 'subreddit', 'thread'])
                        default=['user'])
                        # default=['subreddit'])
                        # default=['thread'])
    parser.add_argument('--out_dir', default='../../data/frequency/')
    # get all metas
    args = parser.parse_args()
    comment_files = args.comment_files
    social_vars = args.social_vars
    out_dir = args.out_dir
    if(comment_files is None):
        all_original_comment_files = get_all_comment_files()
        # get clean/normalized
        all_comment_files = [f.replace('.bz2', '_clean_normalized.bz2') 
                             for f in all_original_comment_files]
        all_meta_files = [f.replace('.bz2', '_clean_normalized_meta.bz2') 
                          for f in all_original_comment_files]
    # social_comment_thresh = 10
    social_comment_thresh = 1
    for social_var in social_vars:
        for comment_file, meta_file in izip(all_comment_files, all_meta_files):
            file_date = re.findall('201[0-9]-[0-9]+', comment_file)[0]
            print('processing comments %s'%(comment_file))
            comment_stats = get_social_var_comment_stats(social_var, comment_file, 
                                                         meta_file, social_comment_thresh)
            # write to file
            out_file = os.path.join(out_dir, '%s_%s_comment_stats.tsv'%(file_date, social_var))
            comment_stats.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
