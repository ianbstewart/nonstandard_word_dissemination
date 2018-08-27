"""
Compute a "decline score" for
each word in vocab based on
the piecewise parameters, 
return all words above the 85% and
write to file for later manual
filtering.
"""
import pandas as pd
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler

def main():
    parser = ArgumentParser()
    parser.add_argument('--piecewise_param_file', default='../../data/frequency/2013_2016_tf_norm_log_2_piecewise_discrete.tsv')
    parser.add_argument('--logistic_param_file', default='../../data/frequency/2013_2016_tf_norm_logistic_params.tsv')
    parser.add_argument('--candidate_out_file', default='../../data/frequency/word_lists/decline_word_candidates.csv')
    args = parser.parse_args()
    piecewise_param_file = args.piecewise_param_file
    logistic_param_file = args.logistic_param_file
    candidate_out_file = args.candidate_out_file
    
    ## load data
    N = 36
    piecewise_params = pd.read_csv(piecewise_param_file, sep='\t', index_col=0, header=0)
    piecewise_params_valid = piecewise_params[(piecewise_params.loc[:, 't'] > 0) & (piecewise_params.loc[:, 't'] < N)]
    logistic_params = pd.read_csv(logistic_param_file, sep='\t', index_col=0, header=0)
    logistic_params_valid = logistic_params[(logistic_params.loc[:, 'loc'] > 0) & (logistic_params.loc[:, 'loc'] < N)]
    
    ## find candidates based on logistic/piecewise scores
    scaler = StandardScaler()
    piecewise_params_valid.loc[:, 'score'] = (piecewise_params_valid.loc[:, 'm1'].apply(lambda x: int(x>0))*piecewise_params_valid.loc[:, 'm2'].apply(lambda x: int(x<0))*piecewise_params_valid.loc[:, 'R2'])
#     piecewise_params_valid.loc[:, 'score'] = scaler.fit_transform(piecewise_params_valid.loc[:, 'score'])
    piecewise_params_valid.sort_values('score', inplace=True, ascending=False)
    piecewise_cutoff_pct = 85
    piecewise_cutoff_score = pd.np.percentile(piecewise_params_valid.loc[:, 'score'], piecewise_cutoff_pct)
    piecewise_score = piecewise_params_valid[piecewise_params_valid.loc[:, 'score'] >= piecewise_cutoff_score]
    piecewise_decline_candidates = piecewise_score.index.tolist()
    logistic_cutoff_pct = 99
    logistic_r2_cutoff = pd.np.percentile(logistic_params_valid.loc[:, 'R2'], logistic_cutoff_pct)
    logistic_params_decline = logistic_params_valid[logistic_params.loc[:, 'R2'] >= logistic_r2_cutoff]
    logistic_decline_candidates = logistic_params_decline.index.tolist()
    
    ## combine and write to file
    combined_score_decline_candidates = set(piecewise_decline_candidates) | set(logistic_decline_candidates)
    piecewise_only = list(set(piecewise_decline_candidates) - set(logistic_decline_candidates))
    logistic_only = list(set(logistic_decline_candidates) - set(piecewise_decline_candidates))
    combined_scores = [piecewise_score.loc[piecewise_only, 'score'],
                       logistic_params_decline.loc[logistic_only, 'R2']]
    combined_scores = pd.DataFrame(pd.concat(combined_scores))
    combined_scores.columns = ['score']
    combined_scores.loc[:, 'word'] = piecewise_only + logistic_only
    combined_scores.loc[:, 'decline_type'] = combined_scores.loc[:, 'word'].apply(lambda x: 'logistic' if x in logistic_only else 'piecewise')
    combined_scores.loc[:, 'standard/proper'] = ''
    combined_scores.sort_values('decline_type', inplace=True, ascending=False)
    combined_scores = combined_scores.loc[:, ['word', 'score', 'decline_type', 'standard/proper']]
    combined_scores.to_csv(candidate_out_file, sep=',', index=False)
    
if __name__ == '__main__':
    main()