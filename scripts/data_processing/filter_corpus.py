"""
Filter non-English subreddits and
spammer users from corpus.
"""
import re, os, codecs, json
import pandas as pd
import argparse
from data_handler import get_default_spammers, get_default_bots, get_non_english_communities
from collections import defaultdict
from bz2 import BZ2File
from itertools import izip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('--out_file', default=None)
    args = parser.parse_args()
    corpus = args.corpus
    out_file = args.out_file
    if(out_file is None):
        out_file = corpus.replace('.bz2', '_filtered.bz2')
    non_english_subs = get_non_english_communities()
    spammers = get_default_spammers()
    bots = get_default_bots()
    spam_users = spammers + bots
    ctr = 0
    with BZ2File(out_file, 'w') as output, BZ2File(corpus, 'r') as text_iter:
        for j in text_iter:
            try:
                j_comment = json.loads(j)
                # check for subreddit/users
                if(j_comment['body'] != '[deleted]' and 
                   j_comment['subreddit'].lower() not in non_english_subs and
                   j_comment['author'].lower() not in spam_users
                   ):
                    output.write(j)
                    ctr += 1
                    if(ctr % 100000 == 0):
                        print('extracted %d normalized comments'%(ctr))
            except Exception, e:
                print('skipped line because error %s with comment %s'%(e, j))
                break

if __name__ == '__main__':
    main()
