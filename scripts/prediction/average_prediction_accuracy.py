"""
Compute average prediction accuracy over all bootstrap iterations
for all models tested with a particular k value.
"""
import pandas as pd
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('accuracy_file')
    args = parser.parse_args()
    accuracy_file = args.accuracy_file
    accuracy_scores = pd.read_csv(accuracy_file, sep='\t', index_col=0)
    # restrict to relevant values
    relevant_cols = ['Accuracy', 'AccuracySD', 'feature_set_name']
    accuracy_scores = accuracy_scores.loc[:, relevant_cols]
    accuracy_score_means = accuracy_scores.groupby('feature_set_name').apply(pd.np.mean)
    accuracy_score_means.loc[:, 'feature_set_name'] = accuracy_score_means.index
    # write to file
    out_file = accuracy_file.replace('.tsv', '_average.tsv')
    accuracy_score_means.to_csv(out_file, sep='\t', index=False)

if __name__ == '__main__':
    main()
