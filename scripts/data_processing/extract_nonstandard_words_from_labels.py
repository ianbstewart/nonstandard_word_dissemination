"""
Extract the nonstandard words from the labelled
word lists => success and failed words.
"""
import pandas as pd
from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    # success
#     parser.add_argument('--word_label_file', default='../../data/frequency/word_lists/2013_2016_success_scores.tsv')
    # fail
    parser.add_argument('--word_label_file', default='../../data/frequency/word_lists/2013_2016_fail_scored_candidate_list.tsv')
    parser.add_argument('--out_dir', default='../../data/frequency/word_lists/')
    args = parser.parse_args()
    word_label_file = args.word_label_file
    out_dir = args.out_dir
    
    ## load data
    # success
#     word_labels = pd.DataFrame(pd.read_csv(word_label_file, sep='\t', index_col=0))
    # fail
    word_labels = pd.read_csv(word_label_file, sep=',', index_col=0, header=None)
    # fail
    word_labels.columns = ['word category']
    word_labels.loc[:, 'fail'] = word_labels.loc[:, 'word category'].apply(lambda x: int(x != 'N' and x!='P' and x != '?'))
    
    ## filter
    # success
#     valid_word_idx = word_labels[word_labels.loc[:, 'success'] == 'Y'].index.tolist()
    # fail
    valid_word_idx = word_labels[word_labels.loc[:, 'fail'] == 1].index.tolist()
    
    ## write to file with category
    valid_word_data = word_labels.loc[valid_word_idx, ['word category']]
    # success
#     out_file = os.path.join(out_dir, '2013_2016_success_words_final.tsv')
    # fail
    out_file = os.path.join(out_dir, '2013_2016_fail_words_final.tsv')
    valid_word_data.to_csv(out_file, sep='\t')
    
if __name__ == '__main__':
    main()