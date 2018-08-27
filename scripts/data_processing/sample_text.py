"""
Reservoir sample a zipped text file. Need this
to train word embeddings in a reasonable time and
to save memory. SORRY BOUT IT.
"""
from bz2 import BZ2File
import argparse
import codecs
import random

def reservoir_sample(in_file, sample_size):
    """
    Get reservoir sample of lines from input file.

    Parameters:
    -----------
    in_file : generator
    Spits out one file at a time.
    sample_size : int
    Sample size.
    
    Returns:
    --------
    sample : [str]
    Sampled lines from file.
    """
    sample = []
    for i, l in enumerate(in_file):
        if(i < sample_size):
            sample.append(l)
        else:
            r = random.randint(0, i)
            if r < sample_size:
                sample[r] = l
        if(i % 100000 == 0):
            print('processed %d lines with sample size %d'%
                  (i+1, len(sample)))
    print('got sample of length %d'%(len(sample)))
    return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_file',
                        default=('/mnt/new_hg190/corpora/reddit_comment_data/'+
                                 'monthly_submission/2015/RC_2015-06_clean_normalized.bz2')
                        )
    parser.add_argument('--sample_size', type=int, default=1000000)
    args = parser.parse_args()
    original_file = args.original_file
    sample_size = args.sample_size
    with BZ2File(original_file, 'r') as in_file:
        in_generator = (l.strip() for l in in_file)
        new_sample = reservoir_sample(in_file, sample_size)
    out_filename = original_file.replace('.bz2', '_sample.txt')
    with codecs.open(out_filename, 'w', encoding='utf-8') as out_file:
        for l in new_sample:
            # try:
            out_file.write(l.decode('utf-8'))
            # except:
            #     print('exception with line %s'%(l))
            #     break

if __name__ == '__main__':
    main()
