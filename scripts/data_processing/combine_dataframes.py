"""
Combine multiple dataframes and re-index accordingly,
then rewrite to file.
"""
import pandas as pd
import argparse

def combine_dataframes(dataframes, axis=0):
    combined = pd.concat(dataframes, axis=axis)
    if(axis == 0):
        combined.index = range(combined.shape[0])
    return combined

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataframes', nargs='+')
    parser.add_argument('out_file')
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--sort_cols', nargs='+', 
                        # default=['word', 'date'])
                        default=None)
    args = parser.parse_args()
    dataframes = args.dataframes
    out_file = args.out_file
    sort_cols = args.sort_cols
    axis = args.axis
    dataframes = [pd.read_csv(d, sep='\t', index_col=0) for d in dataframes]
    combined = combine_dataframes(dataframes, axis=axis)
    print('combined has shape %s'%(str(combined.shape)))
    if(sort_cols is not None):
        combined.sort_values(sort_cols, inplace=True)
    combined.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
