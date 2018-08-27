"""
Generate list of non-English
communities based on random sample
of language-identifications.
"""
import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--community_langs', default='../../data/community_stats/community_lang_sample.tsv')
    args = parser.parse_args()
    community_lang_file = args.community_langs
    community_languages = pd.read_csv(community_lang_file, sep='\t', index_col=0)
    upper_bound = 0.8
    lower_bound = 0.
    english_pcts = community_languages.loc['en'].sort_values(inplace=False, ascending=False)
    non_english_subs = english_pcts[english_pcts < upper_bound][english_pcts > lower_bound].index.tolist()
    print('got non english subs %s'%(str(non_english_subs)))
    out_dir = os.path.dirname(community_lang_file)
    out_fname = os.path.join(out_dir, 'non_english_communities.txt')
    with open(out_fname, 'w') as out_file:
        for s in non_english_subs:
            out_file.write(s+'\n')

if __name__ == '__main__':
    main()
