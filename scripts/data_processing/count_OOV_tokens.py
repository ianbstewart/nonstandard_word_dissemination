"""
Count the total number of OOV tokens in
Reddit corpus and compute % of tokens.
"""
from argparse import ArgumentParser
import os
import re
from functools import reduce
from bz2 import BZ2File
import pandas as pd
from multiprocessing import Pool

# constants
OOV="CHAR-UNK"
OOV_MATCHER=re.compile(OOV)
MAX_PROCESSES=12
TOKENIZE=lambda x: x.split(' ')

def count_tokens(f):
    """
    Count UNK and total tokens in file.
    """
    unk_count = 0
    total_count = 0
    with BZ2File(f) as f_open:
        for i, l in enumerate(f_open):
            l = l.strip()
            l_tokens = TOKENIZE(str(l))
            l_unk_tokens = list(filter(lambda x: OOV_MATCHER.search(x), l_tokens))
            unk_count += len(l_unk_tokens)
            total_count += len(l_tokens)
            if(i % 1000000 == 0):
                print('processed %d lines in %s'%(i, f))
    return f, unk_count, total_count
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='/hg190/corpora/reddit_comment_data/monthly_submission/')
    parser.add_argument('--years', default=[2013, 2014, 2015, 2016])
    parser.add_argument('--out_dir', default='../../output/')
    args = parser.parse_args()
    data_dir = args.data_dir
    years = args.years
    out_dir = args.out_dir
    
    ## collect files
    file_matcher = re.compile('RC_201[0-9]-[01][0-9]_normalized.bz2')
    corpora_dirs = [os.path.join(data_dir, str(y)) for y in years]
    corpora_files = list(reduce(lambda x,y: x+y, [[os.path.join(d, f) for f in os.listdir(d) if file_matcher.search(f) is not None] for d in corpora_dirs]))
    # remove early 2013 stuff
    month_matcher = re.compile('RC_201[0-9]-0[0-5]_normalized.bz2')
    corpora_files = list(filter(lambda x: not (os.path.basename(os.path.dirname(x)) == '2013' and month_matcher.search(x) is not None), corpora_files))
    corpora_files = list(sorted(corpora_files))
    
    ## count!
# serial processing ;_;
#     unk_counts = []
#     total_counts = []
#     for f in corpora_files:
#         print('processing file %s'%(os.path.basename(f)))
#         unk_count, token_count = count_tokens(f)
#         unk_counts.append(unk_count)
#         total_counts.append(total_count)
# parallel processing ^o^
    pool = Pool(MAX_PROCESSES)
    all_counts = pool.map(count_tokens, corpora_files)
    print('finished processing')
    print(all_counts)
    corpora_files, unk_counts, total_counts = zip(*all_counts)
    
    ## compute stats
    total_unk_pct = sum(unk_counts) / sum(total_counts) * 100
    print('total UNK tokens = %.3f percent'%(total_unk_pct))
    
    ## write per-file stats to file
    count_data = pd.np.array([unk_counts, total_counts])
    count_df = pd.DataFrame(count_data)
    count_df.index = ['UNK_tokens', 'tokens']
    # extract year/month
    year_month_matcher = re.compile('(?<=RC_)(201[0-9]-[01][0-9])')
    year_months = [year_month_matcher.search(f).group(0) for f in corpora_files]
    count_df.columns = year_months
    out_file = os.path.join(out_dir, 'UNK_token_stats.tsv')
    count_df.to_csv(out_file, sep='\t', index=True)
    
if __name__ == '__main__':
    main()
