"""
Compute the likelihood of survival for growth-decline words
based on different factor sets.
"""
from __future__ import division
import pandas as pd
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import smooth_stats, get_default_vocab, get_success_fail_words
from argparse import ArgumentParser
import lifelines
from lifelines.utils import k_fold_cross_validation
from itertools import izip
from math import ceil
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
import os
from scipy.stats import ttest_ind

def build_survival_df(death_words, right_censored_words, death_times, covariates, covariate_names):
    """
    Build survival dataframe from the death words, 
    right-censored words, death times and 
    mean covariate values (from $t=0$ to $t=s$, death time).
    
    Parameters:
    -----------
    death_words : [str]
    right_censored_words : [str]
    death_times : pandas.Series
    Death times for all words (by default, right-censored words have a "death" time of N timesteps).
    covariates : [pandas.DataFrame]
    Rows = words, cols = timesteps.
    covariate_names : [str]
    
    Returns:
    --------
    survival_df : pandas.DataFrame
    Rows = words, cols = death time, death (1/0), covariate mean value
    """
    combined_words = death_words + right_censored_words
    N = len(combined_words)
    deaths = pd.Series(pd.np.zeros(N), index=combined_words)
    deaths.loc[death_words] = 1
    survival_df = pd.DataFrame({'t' : death_times, 'death' : deaths})
    print('pre-covariate survival df shape %s'%(str(survival_df.shape)))
    for cov, cov_name in izip(covariates, covariate_names):
        cov_stats = [cov.loc[combined_words[i], :][:death_times[i]].mean() for i in range(N)]
        survival_df.loc[:, cov_name] = cov_stats
        cov_stats = survival_df.loc[:, cov_name]
        if(pd.np.any(pd.np.isnan(cov_stats))):
            cov_stats_nan = cov_stats[cov_stats.isnull()]
            print('nan cov stat %s:\n%s'%(cov_name, cov_stats_nan.head()))
    return survival_df

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/frequency')
    parser.add_argument('--out_dir', default='../../output')
    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    # collect data
    vocab = get_default_vocab()
    tf = pd.read_csv(os.path.join(data_dir, '2013_2016_tf_norm_log.tsv'), sep='\t', index_col=0)
    D_L = pd.read_csv(os.path.join(data_dir, '2013_2016_3gram_residuals.tsv'), sep='\t', index_col=0).loc[vocab, :].fillna(0, inplace=False)
    D_U = pd.read_csv(os.path.join(data_dir, '2013_2016_user_diffusion_log.tsv'), sep='\t', index_col=0).loc[vocab, :].fillna(0, inplace=False)
    D_S = pd.read_csv(os.path.join(data_dir, '2013_2016_subreddit_diffusion_log.tsv'), sep='\t', index_col=0).loc[vocab, :].fillna(0, inplace=False)
    D_T = pd.read_csv(os.path.join(data_dir, '2013_2016_thread_diffusion_log.tsv'), sep='\t', index_col=0).loc[vocab, :].fillna(0, inplace=False)
#     growth_words = get_growth_words()
#     growth_decline_words, split_points = get_growth_decline_words_and_params()
    success_words, fail_words, split_points = get_success_fail_words()
    
    split_points = split_points.apply(lambda x: int(ceil(x)))
    # organize into survival df
    combined_words = fail_words + success_words
    V = len(combined_words)
    deaths = pd.Series(pd.np.zeros(V), index=combined_words)
    deaths.loc[fail_words] = 1
    N = tf.shape[1]
    split_points_combined = pd.concat([split_points, pd.Series([N,] * len(success_words), index=success_words)], axis=0)
    covariates = [tf, D_L, D_U, D_S, D_T]
    covariate_names = ['f', 'D_L', 'D_U', 'D_S', 'D_T']
    survival_df = build_survival_df(fail_words, success_words, split_points_combined, covariates, covariate_names)
    survival_df_nan = survival_df[survival_df.isnull().any(axis=1)]

    # full timeframe test
    # fit regression using all covariates and all data up to and including time of death
    scaler = StandardScaler()
    survival_df_norm = survival_df.copy()
    survival_df_norm[covariate_names] = scaler.fit_transform(survival_df_norm[covariate_names])
    cox_model = CoxPHFitter()
    event_var = 'death'
    time_var = 't'
    cox_model.fit(survival_df_norm, time_var, event_col=event_var)
    regression_output_file = os.path.join(out_dir, 'cox_regression_all_data.txt')
    orig_stdout = sys.stdout
    with open(regression_output_file, 'w') as regression_output:
        sys.stdout = regression_output
        cox_model.print_summary()
        sys.stdout = orig_stdout
    
    # fixed timeframe test
    # fit regression using all covariates and only data up to first m months
    m = 3
    death_words = list(fail_words)
    right_censored_words = list(success_words)
    combined_words = death_words + right_censored_words
    fixed_death_times = pd.Series(pd.np.repeat(m, len(combined_words)), index=combined_words)
    covariates = [tf, D_L, D_U, D_S, D_T]
    covariate_names = ['f', 'D_L', 'D_U', 'D_S', 'D_T']
    survival_df = build_survival_df(death_words, right_censored_words, fixed_death_times, covariates, covariate_names)
    # now provide the actual death/censorship times
    N = tf.shape[1]
    death_times = pd.concat([split_points.loc[death_words], pd.Series([N,]*len(right_censored_words), index=right_censored_words)], axis=0)
    survival_df['t'] = death_times
    cox_model = CoxPHFitter()
    survival_df.loc[:, covariate_names] = scaler.fit_transform(survival_df.loc[:, covariate_names])
    cox_model.fit(survival_df, time_var, event_col=event_var)
    regression_output_file = os.path.join(out_dir, 'cox_regression_first_%d.txt'%(m))
    orig_stdout = sys.stdout
    with open(regression_output_file, 'w') as regression_output:
        sys.stdout = regression_output
        cox_model.print_summary()
        sys.stdout = orig_stdout
    
    # concordance values
    # set up multiple models with different feature sets
    # then run 10-fold cross-validation to generate concordance scores
    # and plot distributions
    cv = 10
    feature_sets = []
    
    covariate_sets = [['f'], ['f', 'D_L'], 
                      ['f', 'D_U', 'D_S', 'D_T'], ['f', 'D_L', 'D_U', 'D_S', 'D_T']]
    covariate_set_names = ['f', 'f+L', 'f+S', 'f+L+S']
    covariate_set_scores = {}
    cv = 10
    for covariate_set, covariate_set_name in izip(covariate_sets, covariate_set_names):
        survival_df_relevant = survival_df.loc[:, covariate_set + [time_var, event_var]]
        cox_model = CoxPHFitter()
        scores = k_fold_cross_validation(cox_model, survival_df_relevant, time_var, event_col=event_var, k=cv)
        covariate_set_scores[covariate_set_name] = scores
    covariate_set_scores = pd.DataFrame(covariate_set_scores).transpose()
    score_names = ['score_%d'%(i) for i in range(cv)]
    covariate_set_scores.columns = score_names
    # significance test between f and f+C, f+D, f+C+D concordance scores
    pval_thresh = 0.05
    baseline_scores = covariate_set_scores.loc['f', score_names]
    covariate_test_names = ['f+L', 'f+S', 'f+L+S']
    # bonferroni correction = alpha / 3
    pval_corrected = pval_thresh / len(covariate_test_names)
    covariate_set_scores.loc[:, 'pval_thresh'] = pval_corrected
    for covariate_test_name in covariate_test_names:
        covariate_test_scores = covariate_set_scores.loc[covariate_test_name, score_names]
        t_stat, pval = ttest_ind(covariate_test_scores, baseline_scores, equal_var=False)
        covariate_set_scores.loc[covariate_test_name, 't_test'] = t_stat
        covariate_set_scores.loc[covariate_test_name, 'pval'] = pval
    # write to file
    out_file = os.path.join(out_dir, 'cox_regression_concordance_%d_fold_scores.tsv'%(cv))
    covariate_set_scores.to_csv(out_file, sep='\t', index=True)

if __name__ == '__main__':
    main()
