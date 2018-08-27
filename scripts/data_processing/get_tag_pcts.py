"""
For all words in vocab, compute percentage
of tokens that were tagged as proper nouns.
"""
from data_handler import get_default_vocab
from argparse import ArgumentParser
from collections import defaultdict
import os, re
import pandas as pd

def get_all_tag_pcts(tag_file):
    """
    Count number of times that each word
    receives any tag.
    
    Parameters:
    -----------
    tag_file : str
    
    Returns:
    --------
    tag_pct : pandas.DataFrame
    Rows = words, cols = tags.
    """
    word_counts = defaultdict(float)
    tag_pcts = defaultdict(lambda : defaultdict(float))
    # cutoff = 1000000
    try:
        for i, l in enumerate(open(tag_file, 'r')):
            l_split = l.strip().split('\t')
            if(len(l_split) > 1):
                w, t, conf = l.strip().split('\t')
                word_counts[w] += 1
                tag_pcts[w][t] += 1
                if(i % 1000000 == 0):
                    print(i)
                # if(i >= cutoff):
                 #    break
    except Exception, e:
        pass
    vocab = tag_pcts.keys()
    for v in vocab:
        for t in tag_pcts[v].keys():
            tag_pcts[v][t] /= word_counts[v]
    tag_pcts = pd.DataFrame(tag_pcts).transpose()
    tag_pcts.fillna(0, inplace=True)
    return tag_pcts

def get_tag_pcts(tag_file, tag='^'):
    """
    Count number of times that each word
    receives the given tag. LESS MEMORY INTENSIVE
    than keeping track of all tags.
    """
    word_counts = defaultdict(float)
    tag_pcts = defaultdict(float)
    # cutoff = 1000000
    try:
        for i, l in enumerate(open(tag_file, 'r')):
            l_split = l.strip().split('\t')
            if(len(l_split) > 1):
                w, t, conf = l.strip().split('\t')
                word_counts[w] += 1
                if(t == tag):
                    tag_pcts[w] += 1
                if(i % 1000000 == 0):
                    print(i)
                # if(i >= cutoff):
                 #    break
    except Exception, e:
        pass
    vocab = tag_pcts.keys()
    for v in vocab:
        tag_pcts[v] /= word_counts[w]
    return tag_pcts

def main():
    parser = ArgumentParser()
    parser.add_argument('tag_file')
    parser.add_argument('--out_dir', default='../../data/frequency/')
    args = parser.parse_args()
    tag_file = args.tag_file
    out_dir = args.out_dir
    # proper_noun_tag = '^'
    # tag_pcts = get_tag_pcts(tag_file, tag=proper_noun_tag)
    tag_pcts = get_all_tag_pcts(tag_file)
    # write to file
    file_date = re.findall('201[0-9]-[0-9]+', tag_file)[0]
    # tag_pcts = pd.DataFrame(pd.Series(tag_pcts), columns=[file_date])
    out_file = os.path.join(out_dir, '%s_tag_pcts.tsv'%(file_date))
    tag_pcts.to_csv(out_file, sep='\t')
        
if __name__ == '__main__':
    main()
