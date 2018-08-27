"""
Remove all OOV words in corpus
(according to externally provided vocab)
and replace with appropriate UNK markers:
1) CAP-UNK (capitalized words)
2) NUM-UNK (numbers)
3) CHAR-UNK (misc.)
"""
import re, os, codecs, json
import pandas as pd
import argparse
from data_handler import get_default_vocab, get_default_stopwords
from collections import defaultdict
from bz2 import BZ2File
from itertools import izip

UNK='UNK'
NUM_UNK='NUM-UNK'
CAP_UNK='CAP-UNK'
CHAR_UNK='CHAR-UNK'
SUB_UNK='SUB'
USER_UNK='USER'
PUNCT=['.', ',', '?', '!', ';', ':',
       '(', ')', '[', ']', '/']
def normalize_corpus(corpus_txt_file, corpus_json_file, meta_file, out_file):
    with BZ2File(out_file, 'w') as output, BZ2File(meta_file, 'w') as meta:
        with BZ2File(corpus_txt_file, 'r') as text_iter:
            with BZ2File(corpus_json_file, 'r') as json_iter:
                for j, l in izip(json_iter, text_iter):
                    l = l.strip()
                    try:
                        j = json.loads(j)
                        l_fixed = []
                        for w in l.split(' '):
                            # TODO: timeit to compare set-inclusion vs. dict lookup
                            w_map = mapper[w.lower()]
                            if(len(w) > 0):
                                if(sub_finder.findall(w)):
                                    w_map = SUB_UNK
                                elif(user_finder.findall(w)):
                                    w_map = USER_UNK
                                elif(w_map == UNK):
                                    if(w[0].isupper() and w.isalpha()):
                                        w_map = CAP_UNK
                                    elif(w.isdigit()):
                                        w_map = NUM_UNK
                                    else:
                                        w_map = CHAR_UNK
                                else:
                                    w_map = w_map.decode('utf-8').lower()
                                l_fixed.append(w_map)
                        # write to file!
                        l_fixed = (' '.join(l_fixed) + '\n').encode('utf-8')
                        output.write(l_fixed)
                        # also get metadata
                        metadata = [str(j['created_utc']), j['author'], j['link_id'], j['subreddit'], j['id'], j['parent_id'], str(j['score'])]
                        meta.write('\t'.join(metadata)+'\n')
                        ctr += 1
                        if(ctr % 100000 == 0):
                            print('extracted %d normalized comments'%(ctr))
                    except Exception, e:
                        print('skipped line because error %s with comment %s'%(e, j))
                        break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
                         # default='/mnt/new_hg190/corpora/reddit_comment_data/monthly_submission/2015/RC-2015-06_clean.txt')
    parser.add_argument('--corpus_json', default=None)
    parser.add_argument('--vocab',
                        default='../../data/frequency/top_100000_vocab.tsv')
    parser.add_argument('--out_file', default=None)
    args = parser.parse_args()
    corpus_file = args.corpus
    corpus_json_file = args.corpus_json
    vocab_file = args.vocab
    out_file = args.out_file
    # try to find default corpus if not provided
    if(corpus_json_file is None):
        corpus_json_file = corpus_txt_file.replace('_clean', '_filtered')
    vocab = pd.read_csv(vocab_file, sep='\t', index_col=0).index.tolist()
    # vocab = get_default_vocab(vocab_file=vocab_file,
    #                           top_k=args.top_k)
    # add stopwords b/c I'm dumb
    stops = get_default_stopwords()
    vocab.extend(stops)
    # add punctuation too
    vocab.extend(PUNCT)
    
    if(out_file is None):
        out_file = corpus.replace('_clean.bz2', '_normalized.bz2')
    mapper = defaultdict(lambda : 'UNK')
    for v in vocab:
        try:
            mapper[v] = v.lower()
        except Exception as e:
            print('problem with word %s'%(v))
    ctr = 0
    sub_finder = re.compile('(?<=r/)\w+')
    user_finder = re.compile('(?<=u/)\w+')
    # also include meta file because duh
    meta_file = out_file.replace('.bz2', '_meta.bz2')
    normalize_corpus(corpus, corpus_json, meta_file, out_file)

if __name__ == '__main__':
    main()
