## Predict success from failed word
## based on first k time steps of
## stats: POS tags alone and frequency.
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

def predict_LR(data, y_var, n_folds):
    """
    Predict success/fail with a given feature set,
    using k months of data.
    
    Parameters:
    -----------
    data_set : list
    y_var : str
    n_folds : int
    
    Returns:
    --------
    results : pandas.DataFrame
    Row = per-fold results, columns = k, AUC, fold number.
    """
    y_val_counts = data.loc[:, y_var].value_counts().sort_values(inplace=False, ascending=True)
    N_minority = y_val_counts.values[0]
    y_vals = data.loc[:, y_var].unique()
    y_idx = []
    for y_val in y_vals:
        y_idx_v = data[data.loc[:, y_var] == y_val].index
        y_idx_v = pd.np.random.choice(y_idx_v, N_minority, replace=False).tolist()
        y_idx += y_idx_v
    data_balanced = data.loc[y_idx, :]
    y_col = data_balanced.loc[:, y_var]
    data_balanced.drop(y_var, axis=1, inplace=True)
    
    # stratified cross-validation
    results = pd.DataFrame()
    results_cols = ['AUC', 'accuracy', 'F1', 'fold']
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=None)
    for i, (train_idx, test_idx) in enumerate(skf.split(data_balanced, y_col)):
        X_train = data_balanced.iloc[train_idx, :]
        y_train = y_col[train_idx]
        X_test = data_balanced.iloc[test_idx, :]
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
    parser.add_argument('--POS_file', default='../../data/frequency/2013_2016_tag_estimates.tsv')
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--out_dir', default='../../output')
    args = parser.parse_args()
    tf_file = args.tf_file
    POS_file = args.POS_file
    k = args.k
    out_dir = args.out_dir
    
    ## load data
    k_range = pd.np.arange(1,k+1)
    tf = pd.read_csv(tf_file, sep='\t', index_col=0).iloc[:, k_range-1]
    tf.columns = map(lambda x: 'f_%d'%(x), k_range)
    POS_tags = pd.read_csv(POS_file, sep='\t', index_col=0).iloc[:, 0]
    # convert to dummy vars
    POS_tags = POS_tags.str.get_dummies()
    POS_tag_list = list(POS_tags.columns)
    # combine
    shared_vocab = tf.index & POS_tags.index
    data = pd.concat([tf.loc[shared_vocab], POS_tags.loc[shared_vocab]], axis=1)
    
    ## restrict to success/fail words
    success_words = get_growth_words()
    fail_words, _ = get_growth_decline_words_and_params()
#     success_words = get_success_words_final()
#     fail_words = get_fail_words_final()
    fail_words = list(set(fail_words))
    # restrict to shared vocab
    success_words = list(set(success_words) & set(shared_vocab))
    fail_words = list(set(fail_words) & set(shared_vocab))
    change_words = success_words + fail_words
    data = data.loc[change_words, :]
    # add success condition
    y_var = 'success'
    data.loc[:, 'success'] = map(lambda x: int(x in success_words), data.index.tolist())
    
    ## organize
    data_sets = [
        data.loc[:, [y_var]+POS_tag_list], # just POS
        data, # f+POS
    ]
    data_set_names = ['POS', 'f+POS']
    results = pd.DataFrame()
    n_folds = 10
    for data_set, data_set_name in izip(data_sets, data_set_names):
        for k in k_range:
            feat_results = predict_LR(data_set, y_var, n_folds)
            feat_results.loc[:, 'k'] = k
            feat_results.loc[:, 'feat_names'] = data_set_name
            results = results.append(feat_results)
    
    ## write to file!!
    k_range_str = '%d_%d'%(min(k_range), max(k_range))
    out_file = os.path.join(out_dir, 'success_%s_window_POS.tsv'%(k_range_str))
    results.to_csv(out_file, sep='\t', index=False)
    
if __name__ == '__main__':
    main()