"""
Compute log of stat from .tsv file then rewrite to new file.
"""
import pandas as pd
from numpy import log10
import argparse

def log_func(s):
    s_min = s[s > 0.].min().min()
    s_log = s.applymap(lambda x: pd.np.log10(x + s_min))
    return s_log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stat_file')
    args = parser.parse_args()
    stat_file = args.stat_file
    stat = pd.read_csv(stat_file, sep='\t', index_col=0)
    # smooth and log
    stat = log_func(stat)
    # write to file
    out_file = stat_file.replace('.tsv', '_log.tsv')
    stat.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
