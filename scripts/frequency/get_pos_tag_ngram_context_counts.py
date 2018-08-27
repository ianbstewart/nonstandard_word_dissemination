"""
Count the number of unique ngram contexts
in which a word occurs.
"""
import pandas as pd
from bz2 import BZ2File
from argparse import ArgumentParser
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_default_vocab
import os, re
from collections import defaultdict, Counter

def get_ngram_counts(tag_file, n):
    tag_queue = []
    word_queue = []
    ctr = 0
    # cutoff = 1000
    END_TAG = 'END'
    START_TAG = 'START'
    BLANK_WORD = 'BLANK'
    # insert START to start
    tag_queue.append(START_TAG)
    word_queue.append(START_TAG)
    ngram_counter = defaultdict(lambda : Counter())
    # cutoff = 100000
    for i, l in enumerate(BZ2File(tag_file, 'r')):
        if(l == '\n'):
            word = END_TAG
            tag = END_TAG
        else:
            word, tag, _ = l.strip().split('\t')
        print(l)
        word_queue.append(word)
        tag_queue.append(tag)
        if(len(tag_queue) >= n):
            for j in range(n):
                word_j = word_queue[j]
                tags_j = [tag_queue[k] for k in range(n) if k != j]
                tag_str = ','.join(tags_j)
                ngram_counter[word_j][tag_str] += 1
            # replace queues
            tag_queue = tag_queue[1:]
            word_queue = word_queue[1:]
        if(l == '\n'):
            word_queue = []
            tag_queue = []
            tag_queue.append(START_TAG)
            word_queue.append(START_TAG)
        # if(i > cutoff):
          #   break
    ngram_counter = pd.DataFrame(ngram_counter).fillna(0, inplace=False).transpose()
    # drop START/END from vocab
    ngram_counter.drop([START_TAG, END_TAG], inplace=True)
    return ngram_counter

def main():
    parser = ArgumentParser()
    parser.add_argument('tag_file')
    parser.add_argument('--out_dir', default='../../data/frequency/')
    parser.add_argument('--n', type=int, default=2)
    args = parser.parse_args()
    tag_file = args.tag_file
    out_dir = args.out_dir
    n = args.n
    ngram_counts = get_ngram_counts(tag_file, n)
    # write to file
    timeframe = re.findall('201[0-9]-[0-9]+', tag_file)[0]
    out_fname = os.path.join(out_dir, '%s_%dgram_tag_counts.tsv'%(timeframe, n))
    ngram_counts.to_csv(out_fname, sep='\t')

if __name__ == '__main__':
    main()
