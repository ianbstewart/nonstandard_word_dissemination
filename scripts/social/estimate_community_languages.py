"""
Estimate the language distribution
in top-500 communities so that we can
throw out the non-English ones. Elitism
but also clean data is good.
"""
import argparse
from langid.langid import classify
from collections import defaultdict
from elasticsearch import Elasticsearch, helpers
from random import random, randint
import os
import pandas as pd

TIMEOUT=600
REPLACE=0.05
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_communities',
                        default='../data/community_stats/2015_2016_top_500_communities.txt')
    parser.add_argument('--years', default='2015')
    parser.add_argument('--sample_size',
                        default=100)
    args = parser.parse_args()
    top_communities = [l.lower().strip() for l in open(args.top_communities, 'r')]
    sample_size = args.sample_size
    years = args.years.split(',')
    lang_ctr = {c : defaultdict(float) for c in top_communities}
    es = Elasticsearch(timeout=TIMEOUT)
    # split over 2 years ;_;
    for i, c in enumerate(top_communities):
        print('querying community #%d = %s'%(i, c))
        # collect sample
        community_post_sample = []
        for y in years:
            # index = 'reddit_comments-%s'%(y)
            index = 'reddit_comments-%s-2'%(y)
            # query = {"query": {"match": {"subreddit": c}}}
            query = {"query": {"constant_score": {"filter": {"term": { "subreddit": c } } }}}
            results = helpers.scan(es, query=query, index=index)
            # reservoir sample
            ctr = 0
            for r in results:
                text = r['_source']['body']
                if(text != '[deleted]'):
                    # print(text)
                    # if smaller than sample, add to sample
                    if(len(community_post_sample) < sample_size):
                        community_post_sample.append(text)
                    # otherwise probabilistically replace
                    elif(random() <= REPLACE):
                        replace_index = randint(0, sample_size-1)
                        community_post_sample[replace_index] = text
                    ctr += 1
                    if(ctr % 100000 == 0):
                        print('processed %d valid comments'%(ctr))
        # convert sample to languages
        for t in community_post_sample:
            language, confidence = classify(t)
            lang_ctr[c][language] += 1
        # normalize!
        for k in lang_ctr[c].keys():
            lang_ctr[c][k] /= sample_size
    # now write to file
    lang_ctr = pd.DataFrame(lang_ctr)
    out_dir = os.path.dirname(args.top_communities)
    out_file = os.path.join(out_dir, 'community_lang_sample.tsv')
    lang_ctr.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
