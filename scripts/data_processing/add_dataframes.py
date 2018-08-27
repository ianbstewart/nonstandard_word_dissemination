"""
Add dataframes together, filling in NaN
values with 0 as necessary, then
rewrite sum to file.
"""
import pandas as pd
from argparse import ArgumentParser

def add_dataframes(df_files, fillna=0.):
    """
    Add dataframes and return sum.
    
    Parameters:
    -----------
    df_files : [str]
    fillna : float
    Optional value to replace NaN values with (default 0).
    Need this to fill index gaps, e.g. if dataframe 1 has different
    index from dataframe 2.
    
    Returns:
    --------
    sum_dataframe : pandas.DataFrame
    """
    sum_dataframe = None
    for df_file in df_files:
        df = pd.read_csv(df_file, sep='\t', index_col=0)
        if(sum_dataframe is None):
            sum_dataframe = df.copy()
        else:
            dfs = [sum_dataframe, df]
            df_idx = [d.index for d in dfs]
            idx_combined = reduce(lambda x,y: x|y, df_idx)
            dfs = [s.loc[idx_combined, :].fillna(fillna, inplace=False) for s in dfs]
            sum_dataframe = reduce(lambda x,y: x+y, dfs)
        print('sum dataframe shape %s'%(str(sum_dataframe.shape)))
    return sum_dataframe

def main():
    parser = ArgumentParser()
    parser.add_argument('df_files', nargs='+')
    parser.add_argument('out_file')
    args = parser.parse_args()
    df_files = args.df_files
    out_file = args.out_file
    sum_dataframe = add_dataframes(df_files)
    # print('sum dataframe shape %s'%(str(sum_dataframe.shape)))
    sum_dataframe.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
