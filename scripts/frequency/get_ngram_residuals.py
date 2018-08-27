"""
Fit a linear regression for log-frequency vs. unique
ngram context counts, then compute the residuals 
for each word.
"""
import pandas as pd
from scipy.stats import linregress
import argparse
import re, os
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import smooth_stats, melt_frames

def get_ngram_residuals_monthly(tf, ngrams, n):
    """
    For each month of data, fit the
    log-f to log-ngram counts and 
    compute the residuals to find words
    that occur in an unusually low or high
    number of contexts.
    
    Parameters:
    -----------
    tf : pandas.DataFrame
    ngrams : pandas.DataFrame
    n : int
    """
    # combine and smooth stats
    all_stats = [tf, ngrams]
    vocab = reduce(lambda x,y: x&y, [s.index for s in all_stats])
    all_stats = [pd.np.log10(smooth_stats(s.loc[vocab, :].fillna(0, inplace=False))) for s in all_stats]
    f_name = 'f'
    ngram_name = 'U_%d'%(n)
    stat_names = [f_name, ngram_name]
    combined_stats = melt_frames(all_stats, stat_names)
    all_resids = []
    for month, group in combined_stats.groupby('date'):
        x = group[f_name]
        y = group[ngram_name]
        m, b, r, p, se = linregress(x, y)
        print('%s = %.3E * tf + %.3E (R=%.3E p=%.3E)'%(ngram_name, m, b, r, p))
        # residual = actual - expected
        expected = m*x + b
        residuals = y - expected
        residuals_name = 'C%d'%(n)
        resids_df = group[['word']]
        resids_df.loc[:, residuals_name] = residuals
        resids_df.loc[:, 'date'] = month
        all_resids.append(resids_df)
    resids_df = pd.concat(all_resids, axis=0)
    combined_stats = pd.merge(combined_stats, resids_df, on=['word', 'date'])
    print('monthly residuals => combined dataframe %s'%(combined_stats.head()))
    residuals = combined_stats[['word', 'date', residuals_name]].pivot_table(index='word', columns='date')
    residuals.columns = residuals.columns.droplevel()
    return residuals

def get_ngram_residuals(tf, ngrams, n):
    """
    Fit log-tf to log-ngram counts and
    compute the residuals to find words that
    occur in an unusually low or high number
    of contexts.
    
    Parameters:
    -----------
    tf : pandas.DataFrame
    ngrams : pandas.DataFrame
    n : int
    
    Return:
    -------
    residuals : pandas.DataFrame
    """
    tf_tab = tf.copy()
    tf_tab['word'] = tf_tab.index
    tf_tab = pd.melt(tf_tab, id_vars=['word'], var_name='date', value_name='f')
    context_tab = ngrams.copy()
    context_tab['word'] = context_tab.index
    ngram_name = '%d'%(n)
    context_tab = pd.melt(context_tab, id_vars=['word'], var_name='date', value_name=ngram_name)
    combined_stats = pd.merge(tf_tab, context_tab, how='inner', on=['word','date'])
    # filter for non-zero counts
    nonzero_stats = combined_stats[(combined_stats[ngram_name] > 0) & (combined_stats['f'] > 0)]
    # smooth and log both stats, then regress
    tf_vals = nonzero_stats['f']
    tf_vals = pd.np.log10(smooth_stats(tf_vals))
    ngram_vals = nonzero_stats[ngram_name]
    ngram_vals = pd.np.log10(smooth_stats(ngram_vals))
    m, b, r, p, se = linregress(tf_vals, ngram_vals)
    print('%s = %.3E * tf + %.3E (R=%.3E p=%.3E)'%(ngram_name, m, b, r, p))
    # residual = actual - expected
    expected = m*tf_vals + b
    residuals = ngram_vals - expected
    residuals_name = 'C%s'%(ngram_name)
    residuals = pd.DataFrame({residuals_name : residuals}, index=nonzero_stats.index)
    combined_stats[residuals_name] = residuals
    # add 0 values for missing context count values
    combined_stats[residuals_name] = combined_stats[residuals_name].fillna(0, inplace=False)
    # reshape to proper dataframe format
    residuals = combined_stats[['word', 'date', residuals_name]].pivot_table(index='word', columns='date')
    residuals.columns = residuals.columns.droplevel()
    return residuals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='../../data/frequency/2015_2016_tf.tsv')
    parser.add_argument('--ngrams', default='../../data/frequency/2015_2016_unique_2gram_counts.tsv')
    args = parser.parse_args()
    tf_file = args.tf
    ngram_file = args.ngrams
    ngram_str = re.findall('[0-9]gram', ngram_file)[0]
    # optional position marker
    if('pos' in ngram_file):
        ngram_str = re.findall('[0-9]gram_[0-9]pos', ngram_file)[0]
    n = int(re.findall('[0-9](?=gram)', ngram_file)[0])
    timeframe = re.findall('201[0-9]_201[0-9]', tf_file)[0]
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    ngrams = pd.read_csv(ngram_file, sep='\t', index_col=0)
    vocab = list(set(tf.index) & set(ngrams.index))
    print('got %d vocab: %s'%(len(vocab), vocab[:10]))
    tf = tf.loc[vocab]
    ngrams = ngrams.loc[vocab]
    print(ngrams.head())
    # check for nans??
    ngrams.fillna(0, inplace=True)
    # residuals = get_ngram_residuals(tf, ngrams, n)
    residuals = get_ngram_residuals_monthly(tf, ngrams, n)
    # combine stats in single dataframe
    out_dir = os.path.dirname(ngram_file)
    out_fname = os.path.join(out_dir, '%s_%s_residuals.tsv'%(timeframe, ngram_str))
    residuals.to_csv(out_fname, sep='\t')

if __name__ == '__main__':
    main()
