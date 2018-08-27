"""
Helper methods to facilitate prediction.
"""
from __future__ import division
from random import random
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import izip, cycle
import pandas as pd
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from collections import defaultdict

def match_words(words, match_candidates, match_stat, k):
    """
    Match each word $w$ with a match $m$ from candidates $C$ 
    based on first $k$ timesteps of data:
    $m = argmin_{c \in C} abs(f_{w,0:k} - f_{c,0:k})$.
    Requires $|words| < |match_candidates|$.
    
    Parameters:
    -----------
    words : [str]
    match_candidates : [str]
    Candidates for matching; $|C| >= |W|$ because it's one-to-one matching.
    match_stat : pandas.DataFrame
    Rows = words, cols = dates.
    k : int
    
    Returns:
    --------
    matched_words : [str]
    """
    match_stats_relevant = match_stat.ix[match_candidates, 0:k]
    matched_words = []
    for i, w in enumerate(words):
        w_stat = match_stat.ix[w, 0:k]
        match = abs(w_stat - match_stats_relevant).sum(axis=1).argmin()
        match_stats_relevant.drop(match, inplace=True)
        matched_words.append(match)
        if(i % 100 == 0):
            print('%d matches made'%(i))
    return matched_words

def match_words_low_diff(words, match_candidates, match_stat, k, upper=None):
    """
    Match words, then filter for word pairs with stat difference
    below upper limit.
    
    Parameters:
    -----------
    words : [str]
    match_candidates : [str]
    Candidates for matching; $|C| >= |W|$ because it's one-to-one matching.
    match_stat : pandas.DataFrame
    Rows = words, cols = dates.
    k : int
    upper : float
    Upper 
    
    Returns:
    --------
    words : [str]
    matched_words : [str]
    match_diffs : pandas.DataFrame
    Rows = pairs, cols = word, match, diff.
    """
    matched_words = match_words(words, match_candidates, match_stat, k)
    match_diffs = abs(match_stat.loc[words, :].iloc[:, 0:k].values - match_stat.loc[matched_words, :].iloc[:, 0:k].values).sum(axis=1)
    match_diffs = pd.DataFrame({'word' : words, 'match' : matched_words, 
                                'diff' : match_diffs})
    if(upper is None or upper < min_diff):
        min_diff = min(match_diffs['diff'])
        upper = min_diff
    match_diffs = match_diffs[match_diffs['diff'] <= upper]
    return match_diffs

def match_word_diffs_all_pairs(words, match_candidates, match_stat, k, min_diff_pct=0.):
    """
    Pair each word with match candidate, 
    then filter for word pairs with stat difference
    below upper limit, where we include all possible low-difference
    pairs in cases where a word has multiple possible pairs.
    
    Parameters:
    -----------
    words : [str]
    match_candidates : [str]
    Candidates for matching; $|C| >= |W|$ because it's one-to-one matching.
    match_stat : pandas.DataFrame
    Rows = words, cols = dates.
    k : int
    min_diff_pct : float
    Upper bound percentile to count as "minimum difference." 
    Default 0 (i.e. absolute minimum).
    
    Returns:
    --------
    match_diffs : pandas.DataFrame
    Rows = pairs, cols = word, match, diff.
    """
    match_stats_relevant = match_stat.loc[match_candidates, :].iloc[:, 0:k]
    match_diffs = defaultdict(list)
    for i, w in enumerate(words):
        w_stat = match_stat.loc[w, :][0:k].values
        match_diffs_w = abs(w_stat - match_stats_relevant).sum(axis=1)
        min_diff = pd.np.percentile(match_diffs_w, min_diff_pct)
        min_matches = match_diffs_w[match_diffs_w <= min_diff].index.tolist()
        for m in min_matches:
            match_diffs['word'].append(w)
            match_diffs['match'].append(m)
            match_diffs['diff'].append(match_diffs_w.loc[m])
        # if(i % 100 == 0):
        #     print('%d matches made'%(i))
    match_diffs = pd.DataFrame(match_diffs)
    return match_diffs

def match_words_split_points(words, match_candidates, match_stat, split_points, k, min_diff_pct, replace=True):
    """
    Match each word with a candidate based on 
    similar match stat values from between
    split point s and prior time s-k.
    
    Parameters:
    -----------
    words : [str]
    match_candidates : [str]
    match_stat : pandas.DataFrame
    split_points : pandas.Series
    k : int
    min_diff_pct : float
    replace : bool
    Replace match candidates after each round of matching.
    
    Returns:
    --------
    match_diffs : pandas.DataFrame
    Rows = pairs, cols = word, match, diff.
    """
    match_stats_relevant = match_stat.loc[match_candidates, :]
    match_diffs = defaultdict(list)
    for i, w in enumerate(words):
        s = split_points.loc[w]
        w_stat = match_stat.loc[w, :][s-k:s].values
        match_diffs_w = abs(w_stat - match_stats_relevant.iloc[:, s-k:s]).sum(axis=1)
        min_diff = pd.np.percentile(match_diffs_w, min_diff_pct)
        min_matches = match_diffs_w[match_diffs_w <= min_diff].index.tolist()
        if(not replace):
            min_matches = min_matches[:1]
        # for every match word m with low difference, add to frame
        for m in min_matches:
            match_diffs['word'].append(w)
            match_diffs['match'].append(m)
            match_diffs['diff'].append(match_diffs_w.loc[m])
        # if no replacement, remove from candidates
        if(not replace):
            match_stats_relevant.drop(min_matches[0], axis=0, inplace=True)
    match_diffs = pd.DataFrame(match_diffs)
    return match_diffs

def get_diff_X_Y_split_points(pos_words, neg_words, split_points, k, feature_set, diff_func=None):
    """
    Compute differenced X and Y stats
    (between split point $s$ and $s-k$)
    by alternating positive and negative words.
    
    Parameters:
    -----------
    pos_words : [str]
    neg_words : [str]
    split_points : pandas.Series
    k : int
    feature_set : [pandas.DataFrame]
    diff_func : func(x,y : z)
    
    Returns:
    --------
    X : pandas.DataFrame
    Rows = samples, cols = differenced feature-timestep stats.
    Y : pandas.Series
    1 = positive, 0 = negative.
    """
    if(diff_func is None):
        diff_func = lambda x,y : x-y
    stat_diff_list = []
    Y = []
    for i, (pos_w, neg_w) in enumerate(izip(pos_words, neg_words)):
        s = split_points.loc[neg_w]
        # need to use s+1 as upper bound because Python indexing is exclusive
        stats_neg_w = pd.np.concatenate([f.loc[neg_w, :][s-k:s+1] for f in feature_set])
        stats_pos_w = pd.np.concatenate([f.loc[pos_w, :][s-k:s+1] for f in feature_set])
        if(i % 2 == 0):
            stats_diff = diff_func(stats_pos_w, stats_neg_w)
            y = 1
        else:
            stats_diff = diff_func(stats_neg_w, stats_pos_w)
            y = 0
        stat_diff_list.append(pd.Series(stats_diff))
        Y.append(y)
    X = pd.concat(stat_diff_list, axis=1).transpose()
    Y = pd.Series(Y)
    return X, Y

def get_stable_candidates_low_var(stat, cutoff_pct=10):
    """
    Generate stable candidate words 
    for matching based on coefficient of 
    variation in stat lower than the 
    provided percentile.
    
    Parameters:
    -----------
    stat : pandas.DataFrame
    cutoff_pct : int
    
    Returns:
    --------
    candidates : [str]
    """
    stat_var = stat.var(axis=1) / stat.mean(axis=1)
    stat_var_cutoff = pd.np.percentile(stat_var, cutoff_pct)
    candidates = stat_var[stat_var < stat_var_cutoff].index.tolist()
    return candidates

def match_filter_candidates(w, candidates, match_stat, match_stat_cutoff, filter_stat, filter_stat_cutoff):
    """
    Get top-K candidates that match w on match_stat, then
    filter those for the top-J filter candidates.
    
    Parameters:
    -----------
    w : str
    candidates : [str]
    match_stat : pandas.Series
    Low difference = good.
    match_stat_cutoff : int
    Number of candidates with lowest match difference. 
    filter_stat : pandas.Series
    Low score = good.
    filter_stat_cutoff : int
    Number of candidates with lowest filter score. 
    
    Returns:
    --------
    matched_candidates : [str]
    """
    match_diff = pd.Series(abs(match_stat.loc[w] - match_stat.loc[candidates])).sort_values(inplace=False, ascending=True)
    filter_candidates = match_diff[1:match_stat_cutoff+1].index
    matched_candidate_scores = filter_stat.loc[filter_candidates].sort_values(inplace=False, ascending=True)[:filter_stat_cutoff]
    matched_candidates = matched_candidate_scores.index.tolist()
    return matched_candidates

def get_differenced_data(pos_words, neg_words, stats, diff_func=None):
    """
    Get differenced data from positive and 
    negative examples according to $diff(stat_p, stat_n)$. 
    Alternate order of positive and negative examples to 
    approximately balance classes (equal numbers of 1 and 0).
    
    Parameters:
    -----------
    pos_words : [str]
    neg_words : [str]
    stats : pandas.DataFrame
    Rows = words, cols = stats.
    diff_func : func
    Difference two vectors (e.g. subtraction).
    
    Returns:
    X : pandas.DataFrame
    Rows = words, cols = stats.
    Y : pandas.Series
    Rows = words.
    """
    if(diff_func is None):
        diff_func = lambda x,y: x-y
    N = int(len(pos_words) / 2)
    pos_neg_pairs = zip(pos_words, neg_words)
    pos_pairs = pos_neg_pairs[:N]
    neg_pairs = pos_neg_pairs[N:]
    pos_words_1, neg_words_1 = map(list, zip(*pos_pairs))
    pos_words_2, neg_words_2 = map(list, zip(*neg_pairs))
    X_pos = diff_func(stats.loc[pos_words_1].values, stats.loc[neg_words_1].values)
    X_neg = diff_func(stats.loc[neg_words_2].values, stats.loc[pos_words_2].values)
    Y = pd.np.array([1,]*len(X_pos) + [0,]*len(X_neg))
    X = pd.np.vstack([X_pos, X_neg])
    X = pd.DataFrame(X, columns=stats.columns)
    # X = pd.concat([X_pos, X_neg], axis=1).transpose()
    Y = pd.Series(Y)
    return X, Y

def get_differenced_data_from_stats(pos_stats, neg_stats, diff_func=None):
    """
    Get differenced data from positive and 
    negative stats according to $diff(stat_p, stat_n)$. 
    Alternate order of positive and negative examples to 
    approximately balance classes (equal numbers of 1 and 0).
    
    Parameters:
    -----------
    pos_stats : pandas.DataFrame
    neg_stats : pandas.DataFrame
    diff_func : func
    Difference two vectors (e.g. subtraction).
    
    Returns:
    X : pandas.DataFrame
    Rows = words, cols = stats.
    Y : pandas.Series
    Rows = words.
    """
    if(diff_func is None):
        diff_func = lambda x,y: x-y
    vocab = pos_stats.index.tolist()
    N = pos_stats.shape[0]
    pos_idx = pd.np.random.choice(vocab, int(N / 2), replace=False).tolist()
    neg_idx = list(set(vocab) - set(pos_idx))
    X_pos = diff_func(pos_stats.loc[pos_idx], neg_stats.loc[pos_idx])
    X_neg = diff_func(neg_stats.loc[neg_idx], pos_stats.loc[neg_idx])
    Y = pd.np.array([1,]*len(X_pos) + [0,]*len(X_neg))
    X = pd.np.vstack([X_pos, X_neg])
    X = pd.DataFrame(X, columns=pos_stats.columns)
    Y = pd.Series(Y)
    return X, Y

def cross_val_test(model, stats, pos_words, neg_words, diff_func, cv=10):
    """
    First computes differenced stats based on positive - negative
    stats, then computes F1 scores from k-fold cross-validation 
    using the given model and stats. 
    
    Parameters:
    -----------
    model : binary classifier
    stats : [pandas.DataFrame]
    pos_words : [str]
    neg_words : [str]
    diff_func : func
    Computes difference between positive and negative stat, 
    e.g. subtraction, division.
    cv : int
    
    Returns:
    --------
    scores = [(float, float)]
    Mean and standard deviation of F1 scores.
    """
    X, Y = get_differenced_data(pos_words, neg_words, stats, diff_func)
    scores = cross_val_score(model, X, Y, cv=cv)
    return scores

def test_feature_sets_all_times(pos_words, neg_words, feature_sets, feature_set_names, n_times, model, diff_func, cv=200):
    """
    Test all feature sets over all time periods.
    
    Parameters:
    -----------
    pos_words : [str]
    Positive class words, i.e. growth.
    neg_words : [str]
    Negative class words, i.e. non-growth.
    feature_sets : [[pandas.DataFrame]]
    All combinations of features to use in prediction.
    feature_set_names : [str]
    Features set names.
    n_times : [int]
    Number of timesteps to include in training.
    model : predictor
    diff_func : func
    Computes difference between positive and negative stat, 
    e.g. subtraction, division.
    
    Returns:
    --------
    score_vals : {str : [(float, float)]}
    Each feature set's list of score means and standard errors.
    """
    score_vals = {}
    all_words = pos_words + neg_words
    for feature_set, feature_set_name in izip(feature_sets, feature_set_names):
        score_vals[feature_set_name] = []
        for n in n_times:
            all_stats = [s.ix[all_words, 0:n] for s in feature_set]
            # combine stats along columns
            all_stats = pd.concat(all_stats, axis=1)
            scores = cross_val_test(model, all_stats, pos_words, neg_words, diff_func, cv=cv)
            scores_mean = scores.mean()
#             scores_err = scores.std() / len(scores)
            scores_err = scores.std()
            score_vals[feature_set_name].append((scores_mean, scores_err))
    return score_vals

def plot_feature_set_scores(score_sets, training_range, title, out_file=None, xlabel='Number of months training', score_name='F1 mean'):
    """
    Plot the prediction score mean and errors
    across all feature sets and all training period
    lengths.
    
    Parameters:
    -----------
    score_sets : {str : {int : (float, float)}}
    Feature set name : { training length : (score mean, score stdev)}.
    training_range : [int]
    title : str
    out_file : str
    Optional file to write plot.
    xlabel : str
    score_name : str
    Optional y axis label for score, default="F1 mean".
    """
    plt.figure(figsize=(10,10))
    cmap = plt.get_cmap('cool')
    markers = cycle(['^', 'o', 's', 'v'])
    linestyles = cycle(['--', '-'])
    ctr = 0
    N = len(score_sets)
    for feature_set_name, score_pairs in score_sets.iteritems():
        # if score pairs are dict of dicts, sort by n vals (the inner dict)
        if(type(score_pairs) is dict or type(score_pairs) is defaultdict):
            score_pairs = sorted(score_pairs.items(), key=lambda x: x[0])
            score_pairs = list(zip(*score_pairs)[1])
        score_means, score_errs = zip(*score_pairs)
        color = cmap(ctr / N)
        linestyle = next(linestyles)
        marker = next(markers)
        plt.plot(training_range, score_means, label=feature_set_name, c=color, linestyle=linestyle, marker=marker)
        plt.errorbar(training_range, score_means, yerr=score_errs, c=color, linestyle=linestyle)
        ctr += 1
    plt.xlabel(xlabel, fontsize=18)
    plt.xlim((min(training_range)-1, max(training_range)+1))
    plt.ylabel(score_name, fontsize=18)
    plt.legend(loc='upper left', prop={'size':14})
    plt.title(title, fontsize=24)
    if(out_file is not None):
        plt.savefig(out_file)
    else:
        plt.show()

def plot_feature_set_scores_df(score_sets, title, out_file=None, xlabel='Number of months training', score_name='F1 mean'):
    """
    Plot the prediction score mean and errors
    across all feature sets and all training period
    lengths.
    
    Parameters:
    -----------
    score_sets : pandas.DataFrame
    Rows = feature set combos, cols = feature set name, k, N, score mean, score err.
    training_range : [int]
    title : str
    out_file : str
    Optional file to write plot.
    xlabel : str
    score_name : str
    Optional y axis label for score, default="F1 mean".
    """
    plt.figure(figsize=(10,10))
    cmap = plt.get_cmap('cool')
    markers = cycle(['^', 'o', 's', 'v'])
    linestyles = cycle(['--', '-'])
    ctr = 0.
    feature_set_names = sorted(score_sets['feature_set'].unique())
    N = len(feature_set_names)
    for feature_set_name, group in score_sets.groupby('feature_set'):
        print('plotting feature set %s'%(feature_set_name))
        group.sort_values('k', inplace=True)
        training_range = group['k']
        score_means = group['score_mean']
        score_errs = group['score_err']
        color = cmap(ctr / N)
        linestyle = next(linestyles)
        marker = next(markers)
        plt.plot(training_range, score_means, label=feature_set_name, c=color, linestyle=linestyle, marker=marker)
        plt.errorbar(training_range, score_means, label='', yerr=score_errs, c=color, linestyle=linestyle)
        ctr += 1
    training_range = sorted(score_sets['k'].unique())
    plt.xlabel(xlabel, fontsize=18)
    plt.xlim((min(training_range)-1, max(training_range)+1))
    plt.ylabel(score_name, fontsize=18)
    plt.legend(loc='upper left', prop={'size':14})
    plt.title(title, fontsize=24)
    if(out_file is not None):
        plt.savefig(out_file)
    else:
        plt.show()

def get_mean_difference_scores(feature_set, n, pos_words, neg_words, diff_func, model, cv=10):
    M = len(feature_set)
    stats = pd.concat([s.ix[:, 0:n] for s in feature_set], axis=1)
    X, Y = get_differenced_data(pos_words, neg_words, stats, diff_func)
    # print(X.iloc[:, 0*n:(0+1)*n].head())
    X_mean = pd.np.hstack([pd.np.mean(X.iloc[:, i*n:(i+1)*n], axis=1).values.reshape(-1,1) for i in range(M)])
    X_mean = MinMaxScaler().fit_transform(X_mean)
    scores = cross_val_score(model, X_mean, Y, cv=cv)
    return scores

def get_non_differenced_stat_prediction_scores(feature_set, n, pos_words, neg_words, model, cv=10):
    """
    Predict pos_words vs. neg_words using raw
    values from the feature set, rather than taking
    the difference between positive and negative features.
    
    Parameters:
    -----------
    feature_set : pandas.DataFrame
    Rows = words, cols = dates.
    n : int
    Number of months to use in training.
    pos_words : [str]
    neg_words : [str]
    model : binary classifier
    cv : int
    
    Returns:
    --------
    scores = numpy.array
    F1 scores of binary classifier performance.
    """
    stats = pd.concat([s.ix[:, 0:n] for s in feature_set], axis=1)
    all_words = pos_words + neg_words
    X = stats.loc[all_words]
    Y = pd.np.array([1,]*len(pos_words) + [0,]*len(neg_words))
    X = MinMaxScaler().fit_transform(X)
    scores = cross_val_score(model, X, Y, cv=cv)
    return scores

def get_differenced_stat_prediction_scores(feature_set, n, pos_words, neg_words, model, cv=10, scoring='f1'):
    """
    Predict pos_words vs. neg_words using difference
    in raw values between pos and neg features.
    
    Parameters:
    -----------
    feature_set : [pandas.DataFrame]
    Rows = words, cols = dates.
    n : int
    Number of months to use in training.
    pos_words : [str]
    neg_words : [str]
    model : binary classifier
    cv : int or k-fold split generator
    scoring : str
    Optional scoring instruction (default F1).
    
    Returns:
    --------
    scores = numpy.array
    F1 scores of binary classifier performance.
    """

    M = len(feature_set)
    stats = pd.concat([s.ix[:, 0:n] for s in feature_set], axis=1)
    # compute each feature's differences
    diff_func = lambda x,y: x-y
    X, Y = get_differenced_data(pos_words, neg_words, stats, diff_func)
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scores = cross_val_score(model, X, Y, cv=cv, scoring=scoring)
    return scores

def get_binary_prediction_scores(feature_set, n, pos_words, neg_words, model, cv=10):
    M = len(feature_set)
    stats = pd.concat([s.ix[:, 0:n] for s in feature_set], axis=1)
    diff_func = lambda x,y: x-y
    X, Y = get_differenced_data(pos_words, neg_words, stats, diff_func)
    X_sum = pd.np.hstack([pd.np.sum(X.ix[:, i*n:(i+1)*n-1], axis=1).values.reshape(-1,1) for i in range(M)])
    X_binary = (X_sum > 0).astype(int)
    X_binary = MinMaxScaler().fit_transform(X_binary)
    scores = cross_val_score(model, X_binary, Y, cv=cv)
    return scores

def get_logit_results(feature_set, feature_names, n, pos_words, neg_words, diff_func=None):
    """
    Fit logistic regression to predict pos_words from
    neg_words according to the mean difference between
    their feature values (up to training month n).
    
    Parameters:
    -----------
    feature_set : [pandas.DataFrame]
    Rows = words, cols = dates.
    feature_names : [str]
    n : int
    pos_words : [str]
    neg_words : [str]
    diff_func : func(x,y : z)
    Compute difference z between vectors x and y.

    Returns:
    --------
    logit_results : statsmodels.discrete.discrete_model.LogitResults
    """
    M = len(feature_set)
    stats = pd.concat([s.ix[:, 0:n] for s in feature_set], axis=1)
    if(diff_func is None):
        diff_func = lambda x,y: x-y
    X, Y = get_differenced_data(pos_words, neg_words, stats, diff_func)
    # mean of differences
    X_mean = pd.np.hstack([pd.np.mean(X.iloc[:, i*n:(i+1)*n-1], axis=1).values.reshape(-1,1) for i in range(M)])
    X = pd.DataFrame(MinMaxScaler().fit_transform(X_mean), columns=feature_names)
    # remove stats with 0 variance
    X = X.ix[:, X.var() > 0.]
    X = add_constant(X)
    logit = Logit(Y, X)
    logit_results = logit.fit()
    return logit_results

def balanced_fold_prediction(match_pair_df, feature_set, k, split_points, model, n=4, min_diff_pct=100):
    """
    Parameters:
    -----------
    match_df : pandas
    feature_set : [pandas.DataFrame]
    k : int
    split_points : pandas.Series
    model : binary classifier
    n : int
    Leave-n-out cross-validation.
    min_diff_pct : float
    
    Returns:
    --------
    scores = [float]
    """
    growth_decline_words = match_pair_df['growth_decline']
    growth_words = match_pair_df['growth']
    match_diffs = match_pair_df['diff']
    min_diff = pd.np.percentile(match_diffs, min_diff_pct)
    match_diffs_valid = match_pair_df[match_pair_df['diff'] <= min_diff]
    growth_words_low_diff = match_diffs_valid['growth'].tolist()
    growth_decline_words_low_diff = match_diffs_valid['growth_decline'].tolist()
    diff_func = lambda x,y : x-y
    stat_diff_list = []
    Y = []
    for i, (g, gd) in enumerate(izip(growth_words_low_diff, growth_decline_words_low_diff)):
        s = split_points.loc[gd]
        stats_gd = pd.np.concatenate([f.loc[gd, :][s-k:s].values for f in feature_set])
        stats_g = pd.np.concatenate([f.loc[g, :][s-k:s].values for f in feature_set])
        if(i % 2 == 0):
            stats_diff = diff_func(stats_g, stats_gd)
            y = 1
        else:
            stats_diff = diff_func(stats_gd, stats_g)
            y = 0
        stat_diff_list.append(pd.Series(stats_diff))
        Y.append(y)
    X = pd.concat(stat_diff_list, axis=1).transpose()
    Y = pd.Series(Y)
    # get balanced cv folds
    cv_folds = []
    pos_indices = Y[Y==1].index.tolist()
    neg_indices = Y[Y==0].index.tolist()
    pd.np.random.shuffle(pos_indices)
    pd.np.random.shuffle(neg_indices)
    N = match_diffs_valid.shape[0]
    cv_range = int(N / 2) - n
    for i in range(cv_range):
        training_indices = (pos_indices[:i] + pos_indices[i+n:] + 
                            neg_indices[:i] + neg_indices[i+n:])
        test_indices = pos_indices[i:i+n] + neg_indices[i:i+n]
        cv_folds.append((training_indices, test_indices))
    scores = cross_val_score(model, X, y=Y, cv=cv_folds, scoring='accuracy')
    return scores
