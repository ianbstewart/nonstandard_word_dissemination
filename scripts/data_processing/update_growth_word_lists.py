"""
Update growth word lists with overlapping vocab
because it's annoying to modify them manually in parallel!
"""
import pandas as pd
from itertools import izip
from argparse import ArgumentParser

def get_overlap_non_nan_vocab(word_dfs, col_names):
    """
    Get overlapping vocabulary across all data
    that doesn't have nan values in the corresponding
    columns.

    Parameters:
    -----------
    word_dfs : [pandas.DataFrame]
    col_names : [str]
    
    Returns:
    --------
    overlap_vocab : [str]
    """
    word_lists = [word_df['word'] for word_df in word_dfs]
    overlap_vocab = reduce(lambda x,y: set(x) & set(y), word_lists)
    print(len(overlap_vocab))
    all_nan_words = set()
    for word_df, col_name in izip(word_dfs, col_names):
        print(col_name)
        nan_index = word_df[col_name].apply(lambda x: type(x) is not str and pd.np.isnan(x))
        nan_words = word_df.ix[nan_index[nan_index].index, 'word']
        print(nan_words)
        all_nan_words.update(nan_words)
    overlap_vocab -= all_nan_words
    overlap_vocab = list(overlap_vocab)
    return overlap_vocab

def main():
    parser = ArgumentParser()
    parser.add_argument('--growth_word_files', nargs='+',
                        default=['../../data/frequency/word_lists/2015_2016_growth_words_clean.csv', '../../data/frequency/word_lists/2015_2016_growth_words_clean_tags.csv', '../../data/frequency/word_lists/2015_2016_growth_words_clean_generation_categories.csv'])
    parser.add_argument('--col_names', nargs='+',
                        default=['word', 'tag', 'category'])
    args = parser.parse_args()
    growth_word_files = args.growth_word_files
    col_names = args.col_names
    growth_word_dfs = [pd.read_csv(f, sep=',', index_col=False) for f in growth_word_files]
    overlap_vocab = get_overlap_non_nan_vocab(growth_word_dfs, col_names)
    print('got %d overlap vocab'%(len(overlap_vocab)))
    for df, fname in izip(growth_word_dfs, growth_word_files):
        df_updated = df['word'].isin(overlap_vocab)
        df_updated = df_updated[df_updated].index
        df_updated = df.ix[df_updated, :]
        df_updated.to_csv(fname, index=False)

if __name__ == '__main__':
    main()
