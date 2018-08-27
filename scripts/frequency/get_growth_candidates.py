"""
Using precomputed growth scores, get candidates for study.
"""
import pandas as pd
from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--growth_score_file', default='../../data/frequency/growth_scores.tsv')
    parser.add_argument('--growth_threshold', type=float, default=85.)
    parser.add_argument('--out_dir', default='../../data/frequency/word_lists/')
    args = parser.parse_args()
    growth_score_file = args.growth_score_file
    growth_threshold = args.growth_threshold
    out_dir = args.out_dir
    
    ## load data
    growth_params = pd.read_csv(growth_score_file, sep='\t', index_col=0)
    # drop nan values
    growth_params = growth_params[growth_params.loc[:, 'spearman'].apply(lambda x: not pd.np.isnan(x))]
    growth_scores = growth_params.loc[:, 'spearman']
    
    ## cutoff candidates
    growth_score_cutoff = pd.np.percentile(growth_scores, growth_threshold)
    candidate_growth_scores = growth_scores[growth_scores >= growth_score_cutoff]
    candidate_growth_scores = pd.DataFrame(candidate_growth_scores)
    candidate_growth_scores.columns = ['score']    
    candidate_growth_scores.loc[:, 'word'] = candidate_growth_scores.index.tolist()
    
    ## write to file; leave blank column for annotation
    candidate_growth_scores.loc[:, 'standard/proper'] = ''
    candidate_growth_scores = candidate_growth_scores.loc[:, ['word', 'score', 'standard/proper']]
    candidate_growth_scores.sort_values('score', inplace=True, ascending=False)
    out_file = os.path.join(out_dir, 'growth_word_candidates.csv')
    candidate_growth_scores.to_csv(out_file, sep=',', index=False)

if __name__ == '__main__':
    main()