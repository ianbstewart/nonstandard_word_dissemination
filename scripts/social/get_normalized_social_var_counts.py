"""
Normalize social var counts (e.g. unique users per word)
by dividing by total var counts at each timestep (e.g. total number of users).
"""
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--social_word_counts', 
                         # default='../data/frequency/2015_2016_user_unique.tsv')
                        default='../data/frequency/2015_2016_subreddit_unique.tsv')
    parser.add_argument('--social_comment_stats',
                        # default='../data/frequency/2015_2016_user_comments.tsv')
                        default='../data/frequency/2015_2016_user_comments.tsv')
    args = parser.parse_args()
    social_word_counts_file = args.social_word_counts
    data_dir = os.path.dirname(social_word_counts_file)
    social_word_counts = pd.read_csv(social_word_counts_file, 
                                     sep='\t', index_col=0)
    social_comment_stats = pd.read_csv(args.social_comment_stats, 
                                       sep='\t', index_col=0)
    # get total unique social vals at each timestep
    unique_social_vals = social_comment_stats.apply(lambda c: len(c[c > 0]),
                                                    axis=0)
    normalized_word_counts = social_word_counts / unique_social_vals
    # write to file
    out_fname = social_word_counts_file.replace('.tsv', '_normalized.tsv')
    normalized_word_counts.to_csv(out_fname, sep='\t')

if __name__ == '__main__':
    main()
