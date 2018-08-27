"""
Get n-grams over corpus, thus
the necessary context for syntactically specific
 words like "af."
"""
import pandas as pd
from data_handler import get_all_comment_files, CommentIter
from frequency_helpers import get_ngram_frequency
from sklearn.feature_extraction.text import CountVectorizer
import os, argparse, codecs, re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', default=5)
    parser.add_argument('--years', default='2015,2016')
    parser.add_argument('--min_df', default=5)
    parser.add_argument('--out_dir', default='../data/frequency/')
    parser.add_argument('--comment_files', default=None, nargs='+')
    parser.add_argument('--keyword', default=None)
    args = parser.parse_args()
    years = args.years.split(',')
    comment_files = args.comment_files
    if(comment_files is None):
        comment_files = get_all_comment_files(years=years)[:1]
    keyword = args.keyword
    # cutoff = pd.np.inf
    # assume comment files are passed in date 
    all_ngram_counts = []
    window = 5
    padding = int(window / 2)
    # matcher = re.compile(
    #     ' '.join(['\S+']*(padding) + [keyword] + ['\S+']*(padding))
    # )
    # print('matcher pattern %s'%(matcher.pattern))
    spacer = re.compile('\s+')
    for comment_file in comment_files:
        comment_date = re.findall('201[0-9]-[0-9]{2}', comment_file)[0]
        print('processing comments %s'%(comment_file))
        # comment_iter = CommentIter(comment_file, 
        #                            stopwords=[],
        #                            return_full_comment=False,
        #                            cutoff=cutoff)
        # joined = (' '.join(c) for c in comment_iter)
        try:
            # only want the keyword and immediate context!
            def gen(in_file):
                ctr = 0
                for l in codecs.open(in_file, 'r', encoding='utf-8'):
                    l = spacer.sub(' ', l.strip().lower())
                    if(l != ''):
                        l = ' '.join(['START']*padding + [l] + ['END']*padding)
                        # print('got clean line %s'%(l))
                        l_split = l.split(' ')
                        if(keyword in l_split):
                            idx = l_split.index(keyword)
                            # print('got keyword index %d'%(idx))
                            relevant = ' '.join([l_split[idx+i] 
                                                 for i in range(-padding, padding+1)])
                            # print('got relevant %s'%(relevant))
                            #try:
                            # relevant = matcher.findall(l)[0]
                            ctr += 1
                            if(ctr % 10000 == 0):
                                print('processed %d lines'%(ctr))
                            # print('got relevant %s'%(relevant))
                            yield relevant
                    #except Exception, e:
                    #    print('bad line %s'%(l))
            # generator = (matchers.findall(('START START ' + l + ' END END'))[0]
            #           for l in codecs.open(comment_file, 'r', encoding='utf-8'))
            generator = gen(comment_file)
            ngram_counts = get_ngram_frequency(generator, stopwords=[], 
                                               min_df=args.min_df, 
                                               ngram_range=(args.window_size,
                                                            args.window_size))
            print('got ngram counts %s'%(ngram_counts))
            # restrict to ngrams containing a keyword
            # if(keyword is not None):
            #     relevant_ngrams = [x for x in ngram_counts.index 
            #                        if len(set(keywords, x.split(' '))) > 0]
            #     ngram_counts = ngram_counts.loc[relevant_ngrams]
            ngram_counts.columns = [comment_date]
            all_ngram_counts.append(ngram_counts)
            # write to file
        except Exception, e:
            print('skipping file %s because error %s'%
                  (comment_file, e))
            break
    all_ngram_counts = pd.concat(all_ngram_counts, axis=1)
    if(keyword is not None):
        comment_filename = '%s_ngrams.tsv'%(keyword)
    else:
        comment_filename = 'all_ngrams.tsv'
    
    out_file = os.path.join(args.out_dir, comment_filename)
    all_ngram_counts.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
