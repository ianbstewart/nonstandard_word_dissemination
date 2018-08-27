## Predict success from failed word
## based on first k time steps of
## stats: frequency, dissemination, etc.
from __future__ import division
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_growth_words, get_growth_decline_words_and_params, get_success_words_final, get_fail_words_final
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.metrics import auc, f1_score
from statsmodels.discrete.discrete_model import Logit
import pandas as pd
from itertools import izip
import os
pd.np.random.seed(123)

def predict_LR(feat_set, feat_name_list, y_var, time_steps, n_folds, use_mean=False):
    """
    Predict success/fail with a given feature set,
    using k months of data.
    
    Parameters:
    -----------
    feat_set : list
    feat_name_list : list
    y_var : str
    k : int
    n_folds : int
    use_mean : bool
    Use the mean over all timesteps for each feature, rather than
    all timesteps separately.
    
    Returns:
    --------
    results : pandas.DataFrame
    Row = per-fold results, columns = k, AUC, fold number.
    """
    # organize
    feat_set = [f.loc[:, [y_var]+time_steps] for f in feat_set]
    # balance data 
    test_feat = feat_set[0]
    y_val_counts = test_feat.loc[:, y_var].value_counts().sort_values(inplace=False, ascending=True)
    N_minority = y_val_counts.values[0]
    y_vals = test_feat.loc[:, y_var].unique()
    y_idx = []
    for y_val in y_vals:
        y_feat = test_feat[test_feat.loc[:, y_var] == y_val]
        y_idx_v = pd.np.random.choice(y_feat.index, N_minority, replace=False).tolist()
        y_idx += y_idx_v
    feat_set_balanced = []
    for f in feat_set:
        f_balanced = f.loc[y_idx, :]
        feat_set_balanced.append(f_balanced)
    
    # combine data with appropriate col names
    feat_set_combined = []
    y_col = feat_set_balanced[0].loc[:, y_var]
    for f, feat_name in izip(feat_set_balanced, feat_name_list):
        f_cols = filter(lambda x: x!=y_var, f.columns)
        # optional mean
        if(use_mean):
            f_fixed = pd.DataFrame(f.loc[:, f_cols].mean(axis=1), columns=[feat_name])
        else:
            f_columns_fixed = map(lambda c: '%s_%s'%(feat_name, c), f_cols)
            f_fixed = f.rename(columns=dict(zip(f_cols, f_columns_fixed))).loc[:, f_columns_fixed]
        # rescale
        f_fixed = pd.DataFrame(scale(f_fixed), index=f_fixed.index, columns=f_fixed.columns)
        feat_set_combined.append(f_fixed)
    feat_set_combined = pd.concat(feat_set_combined, axis=1)
    
    # stratified cross-validation
    results = pd.DataFrame()
    results_cols = ['AUC', 'accuracy', 'F1', 'fold']
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=None)
    for i, (train_idx, test_idx) in enumerate(skf.split(feat_set_combined, y_col)):
        X_train = feat_set_combined.iloc[train_idx, :]
        y_train = y_col[train_idx]
        X_test = feat_set_combined.iloc[test_idx, :]
        y_test = y_col[test_idx]
        lr = LogisticRegression(fit_intercept=True)
        lr.fit(X_train, y_train)
#         if(i == 0):
#             # do test logistic regression for sanity check
#             print(X_train.head())
#             print(X_train.mean(axis=0))
#             logit = Logit(y_train, X_train).fit()
#             print('feature weights')
#             print(logit.summary())
        y_prob = pd.np.array(lr.predict_proba(X_test))[:, 1]
        y_pred = lr.predict(X_test)
        # get AUC
        test_auc = auc(y_test, y_prob)
        # get F1
        test_F1 = f1_score(y_test, y_pred)
        test_accuracy = sum(y_test == y_pred) / len(y_pred)
        results = results.append(pd.Series([test_auc, test_accuracy, test_F1, i]), ignore_index=True)
    results.columns = results_cols
    return results

def main():
    parser = ArgumentParser()
    parser.add_argument('--tf_file', default='../../data/frequency/2013_2016_tf_norm_log.tsv')
    parser.add_argument('--DL_file', default='../../data/frequency/2013_2016_3gram_residuals.tsv')
    parser.add_argument('--DU_file', default='../../data/frequency/2013_2016_user_diffusion.tsv')
    parser.add_argument('--DS_file', default='../../data/frequency/2013_2016_subreddit_diffusion.tsv')
    parser.add_argument('--DT_file', default='../../data/frequency/2013_2016_thread_diffusion.tsv')
    parser.add_argument('--k', default=12, type=int)
    parser.add_argument('--out_dir', default='../../output')
    args = parser.parse_args()
    tf_file = args.tf_file
    DL_file = args.DL_file
    DU_file = args.DU_file
    DS_file = args.DS_file
    DT_file = args.DT_file
    k = args.k
    out_dir = args.out_dir
    
    ## load data
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    DL = pd.read_csv(DL_file, sep='\t', index_col=0)
    DU = pd.read_csv(DU_file, sep='\t', index_col=0)
    DS = pd.read_csv(DS_file, sep='\t', index_col=0)
    DT = pd.read_csv(DT_file, sep='\t', index_col=0)
    all_stats = [tf, DL, DU, DS, DT]
    all_stats = [s.fillna(0, inplace=False) for s in all_stats]
    shared_vocab = list(reduce(lambda x,y : x&y, map(lambda y: y.index, all_stats)))
    k_start = 0
    k_range = range(k_start+1,k_start+k+1)
    n_folds = 10
    all_time_steps = tf.columns.tolist()
    
    ## restrict to success/fail words
    success_words = get_growth_words()
    fail_words, _ = get_growth_decline_words_and_params()
    # restrict to shared vocab
    success_words = list(set(success_words) & set(shared_vocab))
    fail_words = list(set(fail_words) & set(shared_vocab))
    change_words = success_words + fail_words
    
    all_stats = [s.loc[change_words, :] for s in all_stats]
    # add success condition
    y_var = 'success'
    for s in all_stats:
        s.loc[:, y_var] = map(lambda x: int(x in success_words), s.index.tolist())
    
    ## organize
    feat_sets = [
        [all_stats[0]], 
        [all_stats[0], all_stats[1]], 
        [all_stats[0], all_stats[2], all_stats[3], all_stats[4]], 
        all_stats
    ]
    feat_name_lists = [
        ['f'], 
        ['f','DL'], 
        ['f','DU','DS','DT'], 
        ['f','DL','DU','DS','DT']
    ]
    feat_set_names = ['f', 'f+L', 'f+S', 'f+L+S']
    results = pd.DataFrame()
    use_mean = False
    for feat_set, feat_set_name, feat_name_list in izip(feat_sets, feat_set_names, feat_name_lists):
        for k_ in k_range:
            time_steps = all_time_steps[k_start:k_]
            feat_results = predict_LR(feat_set, feat_name_list, y_var, time_steps, n_folds, use_mean=use_mean)
            feat_results.loc[:, 'k'] = k_
            feat_results.loc[:, 'feat_names'] = feat_set_name
            results = results.append(feat_results)
    
    ## write to file!!
    k_range_str = '%s_%s'%(min(k_range), max(k_range))
    if(use_mean):
        out_file = os.path.join(out_dir, 'success_%s_window_mean.tsv'%(k_range_str))
    else:
        out_file = os.path.join(out_dir, 'success_%s_window.tsv'%(k_range_str))
    results.to_csv(out_file, sep='\t', index=False)
    
if __name__ == '__main__':
    main()