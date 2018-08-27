"""
Extract top k communities from
count file and write to line-separated
.txt file for later processing.
"""
import pandas as pd
from data_handler import get_default_communities
import argparse, os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', 
                        default='../data/community_stats/')
    parser.add_argument('--k', default=500)
    args = parser.parse_args()
    start_index = 0
    k = args.k
    communities = get_default_communities(start_index=start_index,
                                          k=k)
    years = ['2015', '2016']
    out_file = os.path.join(args.out_dir, 
                            '%s_top_%d_communities.txt'%
                            ('_'.join(years), k))
    with open(out_file, 'w') as output:
        for c in communities:
            output.write('%s\n'%(c))
