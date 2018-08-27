"""
Compute POS tag estimates by computing argmax over tag percents.
"""
import pandas as pd
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('tag_pcts')
    parser.add_argument('out_file')
    args = parser.parse_args()
    tag_pct_file = args.tag_pcts
    out_file = args.out_file
    tag_pcts = pd.read_csv(tag_pct_file, sep='\t', index_col=0)
    # drop proper nouns first
    PPN = '^'
    tag_estimates = pd.DataFrame(tag_pcts.drop(PPN, axis=1, inplace=False).apply(lambda r: r.argmax(), axis=1), columns=['POS'])
    # write to file
    tag_estimates.to_csv(out_file, sep='\t')
    
if __name__ == '__main__':
    main()