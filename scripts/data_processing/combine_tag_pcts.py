"""
Average the tag percents across multiple months.
"""
from __future__ import division
import pandas as pd
from argparse import ArgumentParser
from data_handler import get_mean_df, get_default_vocab
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('tag_pcts', nargs='+')
    parser.add_argument('--timeframe', default='2013_2016')
    args = parser.parse_args()
    tag_pct_files = args.tag_pcts
    timeframe = args.timeframe
    vocab = get_default_vocab()
    tag_pct_df_list = [pd.read_csv(f, sep='\t', index_col=0).loc[vocab] for f in tag_pct_files]
    tag_pct_mean = get_mean_df(tag_pct_df_list)
    print('tag pct mean has shape %s'%(str(tag_pct_mean.shape)))
    print(tag_pct_mean.head())
    # write to file
    out_dir = os.path.dirname(tag_pct_files[0])
    out_file = os.path.join(out_dir, '%s_tag_pcts.tsv'%(timeframe))
    tag_pct_mean.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
