"""
1-way ANOVA on DC values grouped by major parts-of-speech.
"""
import pandas as pd
from scipy.stats import f_oneway
from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('tag_estimates')
    parser.add_argument('dependent_var')
    parser.add_argument('dependent_var_name')
    parser.add_argument('out_dir')
    parser.add_argument('--tags_to_test', nargs='+', default=None)
    args = parser.parse_args()
    tag_estimate_file = args.tag_estimates
    dependent_var_file = args.dependent_var
    dependent_var_name = args.dependent_var_name
    out_dir = args.out_dir
    tags_to_test = args.tags_to_test
    # load data
    tag_estimates = pd.read_csv(tag_estimate_file, sep='\t', index_col=0)
    dependent_var = pd.read_csv(dependent_var_file, sep='\t', index_col=0).fillna(0, inplace=False)
    dependent_var_mean = dependent_var.mean(axis=1)
    vocab = set(tag_estimates.index) & set(dependent_var_mean.index)
    tag_estimates = tag_estimates.loc[vocab]
    if(tags_to_test is None):
        tags_to_test = tag_estimates.loc[:, 'POS'].unique().tolist()
    # group by POS 
    tag_groups = [(t, group.index.tolist()) for t, group in tag_estimates.groupby('POS') if t in tags_to_test]
    dependent_var_groups = [(t, dependent_var_mean.loc[g].tolist()) for t, g in tag_groups]
#     print(dependent_var_groups[0][:10])
    # test for difference in means
    f, pval = f_oneway(*zip(*dependent_var_groups)[1])
    test_results = pd.DataFrame(pd.Series([f, pval], index=['f', 'pval'])).transpose()
    # write to file!!
    out_file = os.path.join(out_dir, 'tag_vs_%s_ANOVA.tsv'%(dependent_var_name))
    test_results.to_csv(out_file, sep='\t', float_format='%.3E')
    
if __name__ == '__main__':
    main()