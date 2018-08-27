"""
Combine list of tag ngram count files
into matrix of unique tag ngrams
per word per timestep.
"""
import pandas as pd
from argparse import ArgumentParser
import re, os

def main():
    parser = ArgumentParser()
    parser.add_argument('ngram_tag_count_files', nargs='+')
    parser.add_argument('--timeframe', default='2013_2016')
    args = parser.parse_args()
    tag_files = sorted(args.ngram_tag_count_files)
    timeframe = args.timeframe
    unique_ngram_count_list = []
    for tag_file in tag_files:
        tag_counts = pd.read_csv(tag_file, sep='\t', index_col=0)
        tag_file_timeframe = re.findall('201[0-9]-[01][0-9]', tag_file)[0]
        unique_ngram_counts = tag_counts.apply(lambda x: len(x[x>0]), axis=1)
        unique_ngram_counts = pd.DataFrame(unique_ngram_counts, columns=[tag_file_timeframe])
        unique_ngram_count_list.append(unique_ngram_counts)
    # combine counts, write to file
    unique_ngram_counts = pd.concat(unique_ngram_count_list, axis=1)
    unique_ngram_counts = unique_ngram_counts.loc[:, sorted(unique_ngram_counts.columns)]
    out_dir = os.path.dirname(tag_files[0])
    n = int(re.findall('[0-9](?=gram)', tag_files[0])[0])
    out_fname = os.path.join(out_dir, '%s_unique_%dgram_tag_counts.tsv'%(timeframe, n))
    unique_ngram_counts.to_csv(out_fname, sep='\t')

if __name__ == '__main__':
    main()
