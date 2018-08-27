"""
Combine separate monthly frequency
data files into one hideous file
for easy time series analysis.
"""
import pandas as pd
import os, re
import argparse
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import combine_data_files

def main():
    """
    Read frequency files, combine, then
    rewrite to file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--combined_name',
                        # default='2015_2016_tf')
                        default='2014_2015_tf')
    parser.add_argument('--data_dir',
                        default='../../data/frequency/')
    parser.add_argument('--tf_files', nargs='+',
                        default=None)
    parser.add_argument('--min_tf', type=int, default=5)
    args = parser.parse_args()
    combined_name = args.combined_name
    data_dir = args.data_dir
    tf_files = args.tf_files
    min_tf = args.min_tf
    out_dir = data_dir
    if(tf_files is None):
        date_match_str = '201[0-9]-[0-9]{2}'
        file_matcher = re.compile('%s.tsv'%(date_match_str))
        date_matcher = re.compile(date_match_str)
        combine_data_files(combined_name, data_dir, 
                           file_matcher, date_matcher,
                           out_dir)
    else:
        # all_data = {}
        all_data = []
        for f in tf_files:
            print('reading %s'%(f))
            d = re.findall('201[0-9]-[01][0-9]', f)[0]
            # tf = pd.read_csv(f, sep='\t', index_col=0)
            # dataframe takes too much space...just use a dictionary
            tf = {}
            with open(f, 'r') as tf_iter:
                for i, l in enumerate(tf_iter):
                    if(i > 0):
                        w, count = l.split('\t')
                        count = int(count)
                        tf[w] = count
            print('got %d counts'%(len(tf)))
            # get rid of non-alpha words
            alpha_words = list(filter(lambda x: type(x) is str and x.isalpha(), tf.keys()))
            tf = {k : tf[k] for k in alpha_words}
            print('%d alpha words'%(len(alpha_words)))
            # get rid of low tf words
            tf = {k : v for k,v in tf.iteritems() if v > min_tf}
            print('%d words at min tf %d'%(len(tf), min_tf))
            tf = pd.DataFrame(tf, index=[d]).transpose()
            print('got df %s'%(str(tf.shape)))
            all_data.append(tf)
            # date_str = re.findall('201[456]-[0-9]+', f)[0]
            # all_data[date_str] = tf.ix[:, 0]
            # all_data = all_data.join(tf, how='outer')
            # print('full data has shape %s'%(str(all_data.shape)))
        # all_data = pd.DataFrame(all_data)
        # all_data = [pd.read_csv(f, sep='\t', index_col=0) for f in tf_files]
        all_data = pd.concat(all_data, axis=1)
        print('full data has shape %s'%(str(all_data.shape)))
        out_fname = os.path.join(out_dir, '%s.tsv'%(combined_name))
        all_data.to_csv(out_fname, sep='\t')

if __name__ == '__main__':
    main()
