"""
Significance testing between concordance scores
from different feature sets.
"""
import pandas as pd
from scipy.stats import ttest_ind
from argparse import ArgumentParser
import os

def main():
  parser = ArgumentParser()
  parser.add_argument('--concordance_scores', default='../../output/results/cox_regression_concordance_10_fold_scores.tsv')
  parser.add_argument('--out_dir', default='../../output/results/')
  args = parser.parse_args()
  concordance_score_file = args.concordance_scores
  out_dir = args.out_dir
  # load data
  concordance_scores = pd.read_csv(concordance_score_file, sep='\t', index_col=0)
  # compare all possible combos
  feature_set_names = concordance_scores.index.tolist()
  score_names = [c for c in concordance_scores if 'score' in c]
  feature_set_scores = [concordance_scores.loc[f, score_names] for f in feature_set_names]
  F = len(feature_set_names)
  test_results = []
  for i in range(F):
      feature_set_score_i = feature_set_scores[i]
      feature_set_name_i = feature_set_names[i]
      for j in range(i+1, F):
          feature_set_score_j = feature_set_scores[j]
          feature_set_name_j = feature_set_names[j]
          stat, pval = ttest_ind(feature_set_score_i, feature_set_score_j)
          test_results.append([feature_set_name_i, feature_set_name_j, stat, pval])
  test_results = pd.DataFrame(test_results, columns=['feature_set_1', 'feature_set_2', 'stat', 'pval'])
  # write to file
  out_file = os.path.join(out_dir, 'concordance_score_feature_set_comparison_tests.tsv')
  test_results.to_csv(out_file, sep='\t')
  
if __name__ == '__main__':
  main()