"""
Compute average value of each cell across all dataframes.
"""
from argparse import ArgumentParser
import os
import pandas as pd

def average_dataframes(dataframes):
    """
    Compute average of all cells in all dataframes.

    Parameters:
    -----------
    dataframes : [pandas.DataFrame]
    
    Return:
    -------
    average_dataframe : pandas.DataFrame
    """
    combined_dataframe = pd.concat(dataframes, axis=0)
    # now get average
    average_dataframe = combined_dataframe.groupby(combined_dataframe.index).mean()
    return average_dataframe

def main():
    parser = ArgumentParser()
    parser.add_argument('dataframes', nargs='+')
    parser.add_argument('--data_name', default='tag_pcts')
    parser.add_argument('--timeframe', default='2015_2016')
    args = parser.parse_args()
    dataframe_files = args.dataframes
    print(dataframe_files)
    timeframe = args.timeframe
    data_name = args.data_name
    dataframes = [pd.read_csv(d, sep='\t', index_col=0) for d in dataframe_files]
    average_dataframe = average_dataframes(dataframes)
    # write to file
    out_dir = os.path.dirname(dataframe_files[0])
    out_fname = os.path.join(out_dir, '%s_%s.tsv'%(timeframe, data_name))
    average_dataframe.to_csv(out_fname, sep='\t')

if __name__ == '__main__':
    main()
