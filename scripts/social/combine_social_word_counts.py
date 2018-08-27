"""
Combine the separate social word count files.
"""
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import combine_data_files
import argparse
import os, re
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/frequency/')
    parser.add_argument('--social_vars', nargs='+', 
                        #default=['user', 'thread', 'subreddit'])
                        default=['user'])
                        # default=['subreddit'])
                        # default=['thread'])
    parser.add_argument('--timeframe', default='2015_2016')
    args = parser.parse_args()
    data_dir = args.data_dir
    social_vars = args.social_vars
    timeframe = args.timeframe
    for social_var in social_vars:
        all_social_word_counts = []
        social_files = [os.path.join(data_dir, f) 
                        for f in os.listdir(data_dir) 
                        if re.findall('201[0-9]-[0-9]{2}_%s_unique'%(social_var), f)]
        social_files = sorted(social_files)
        for f in social_files:
            print('processing file %s'%(f))
            file_date = re.findall('201[0-9]-[0-9]+', f)[0]
            social_word_counts = {file_date : []}
            vocab = []
            for l in open(f, 'r'):
                word, count = l.strip().split('\t')
                social_word_counts[file_date].append(float(count))
                vocab.append(word)
            social_word_counts = pd.DataFrame(social_word_counts, index=vocab)
            print('extracted social counts of shape %s'%(str(social_word_counts.shape)))
            all_social_word_counts.append(social_word_counts)
            
        all_social_word_counts = pd.concat(all_social_word_counts, axis=1)
        # write to file!
        out_fname = os.path.join(data_dir, '%s_%s_unique.tsv'%(timeframe, social_var))
        all_social_word_counts.to_csv(out_fname, sep='\t')

if __name__ == '__main__':
    main()
