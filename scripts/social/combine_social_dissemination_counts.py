"""
Combine separate social diffusion files.
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
    parser.add_argument('--social_var', nargs='+', 
                        # default='user')
                        default='subreddit')
                        # default='thread')
    parser.add_argument('--timeframe', default='2013_2016')
    args = parser.parse_args()
    data_dir = args.data_dir
    social_var = args.social_var
    timeframe = args.timeframe
    all_social_diffusion = []
    social_files = [os.path.join(data_dir, f) 
                    for f in os.listdir(data_dir) 
                    if re.findall('^201[0-9]-[0-9]{2}_%s_diffusion.tsv'%(social_var), f)]
    social_files = sorted(social_files)
    for f in social_files:
        print('processing file %s'%(f))
        file_date = re.findall('201[0-9]-[0-9]+', f)[0]
        social_diffusion = {file_date : []}
        vocab = []
        social_diffusion = pd.read_csv(f, sep='\t', index_col=0)
        print('extracted social counts of shape %s'%(str(social_diffusion.shape)))
        all_social_diffusion.append(social_diffusion)

    all_social_diffusion = pd.concat(all_social_diffusion, axis=1)
    # write to file!
    out_fname = os.path.join(data_dir, '%s_%s_diffusion.tsv'%(timeframe, social_var))
    all_social_diffusion.to_csv(out_fname, sep='\t')

if __name__ == '__main__':
    main()
