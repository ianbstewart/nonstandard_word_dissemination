"""
Extract diffusion stats from the raw
data rather than preprocessed stats...
this is not a dumb idea.
"""
from itertools import izip
from bz2 import BZ2File
from collections import Counter, defaultdict
import argparse
import re, os
import pandas as pd
from math import factorial

def get_diffusion_data(data_file, meta_file, social_var='user', vocab=None):
    """
    Count words per social var and social vars
    per word.
    
    Parameters:
    -----------
    data_file : str
    meta_file : str
    social_var : str
    user, subreddit, thread
    vocab : list
    Optional list of vocab to include.

    Returns:
    --------
    m_i : pandas.Series
    Words per social var.
    u_i : pandas.Series
    Social vars per word.
    """
    # words per user
    m_i = Counter()
    # social vars per word
    u_i = defaultdict(set)
    # word counts
    f_w = Counter()
    meta_indices = {'user' : 1, 'thread' : 2, 'subreddit' : 3}
    meta_index = meta_indices[social_var]
    # for sampling
#    meta_out = meta_file.replace('.bz2', 'sample.bz2')
#    data_out = data_file.replace('.bz2', 'sample.bz2')
    cutoff = 6e7
    with BZ2File(meta_file, 'r') as meta, BZ2File(data_file, 'r') as data:
        # with BZ2File(meta_out, 'w') as meta_output, BZ2File(data_out, 'w') as data_output:
        for ctr, (m, d) in enumerate(izip(meta, data)):
                # meta_output.write(m)
                # data_output.write(d)
            var = m.strip().split('\t')[meta_index].lower()
            words = d.strip().split()
            if(vocab is not None):
                words = set(words) & vocab
            # print('got var %s'%(var))
            # print('got words %s'%(str(words)))
            m_i[var] += len(words)
            for w in words:
                u_i[w].add(var)
                f_w[w] += 1
            ctr += 1
            # if(ctr % 1000 == 0):
            #     print('got %d comments'%(ctr))
            if(ctr >= cutoff):
                break
    # change u_i to counts
    u_i = {w : len(social_vars) for w, social_vars in u_i.iteritems()}
    m_i = pd.Series(m_i)
    u_i = pd.Series(u_i)
    f_w = pd.Series(f_w)
    # normalize frequency
    f_w /= f_w.sum()
    return m_i, u_i, f_w
    # return m_i, u_i

def get_expected_exact(f_w, m_i):
    """
    Computed expected social var count for each
    word in frequency vocabulary.
    U_expected = sum_i ( 1 - exp( -f_w * u_i ) )
    
    Parameters:
    -----------
    f_w : pandas.Series
    Normalized frequency per word in vocabulary.
    m_i : pandas.Series
    Number of tokens per unique social var.

    Returns:
    --------
    expected : pandas.Series
    Number of expected social vars per word in vocabulary.
    """
    # expected = (1. - pd.np.exp(-pd.np.outer(m_i,f_w))).sum(axis=0)
    vocab = f_w.index.tolist()
    expected = {}
    for i, v in enumerate(vocab):
        expected[v] = (1. - pd.np.exp(-f_w.loc[v]*m_i)).sum()
        if(i % 1000 == 0):
            print('processed %d vocab'%(i))
    expected = pd.Series(expected)
    return expected

def get_expected_approx(f_w, m_i, approx_n=20):
    # m1 = m_i.sum()
    # m2 = (m_i ** 2).sum()
    # m3 = (m_i ** 3).sum()
    # m4 = (m_i ** 4).sum()
    # m5 = (m_i ** 5).sum()
    expected = 0
    for n in range(1, approx_n+1):
        expected += ((-1.) ** (n-1)) * (1. / factorial(n)) * (f_w ** n) * (m_i ** n).sum()
    # expected = f_w * m1 - .5 * (f_w ** 2) * m2 + (1./6) * (f_w ** 3) * m3 
    # expected = f_w * m1 - .5 * (f_w ** 2) * m2 + (1./6) * (f_w ** 3) * m3 - (1./24) * (f_w ** 4) * m4 + (1./125) * (f_w ** 5) * m5
    return expected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('comment_file')
    parser.add_argument('--social_var', default='user')
    parser.add_argument('--tf_file', default='../../data/frequency/2015_2016_tf_norm.tsv')
    parser.add_argument('--vocab', default=None)
    args = parser.parse_args()
    comment_file = args.comment_file
    social_var = args.social_var
    tf_file = args.tf_file
    vocab_file = args.vocab
    date_str = re.findall('201[0-9]-[0-9]+', comment_file)[0]
    print('got date %s'%(date_str))
    meta_file = comment_file.replace('.bz2', '_meta.bz2')
    if(vocab_file is not None):
        vocab = set(pd.read_csv(vocab_file, sep='\t', index_col=0).index)
    else:
        vocab = None
    m_i, u_i, tf = get_diffusion_data(comment_file, meta_file, social_var=social_var, vocab=vocab)
    u_i.sort_values(inplace=True, ascending=False)
    m_i.sort_values(inplace=True, ascending=False)
    tf.sort_values(inplace=True, ascending=False)
    print('u_i %s'%(u_i))
    print('m_i %s'%(m_i))
    print('tf %s'%(tf))
    # write to file??
    u_out = os.path.join(os.path.dirname(tf_file), '%s_unique_%s_counts.tsv'%(date_str, social_var))
    u_i.to_csv(u_out, sep='\t')
    # print('got date str %s'%(date_str))
    # tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    # vocab = u_i.index.tolist()
    # tf = tf.ix[vocab, date_str]
    # get expected counts and diffusion
    # expected = get_expected_approx(tf, m_i)
    expected = get_expected_exact(tf, m_i)
    expected.sort_values(inplace=True, ascending=False)
    print('got expected %s'%(expected))
    diffusion = u_i / expected
    diffusion.sort_values(inplace=True, ascending=False)
    print('got diffusion %s'%(diffusion))
    # write to file
    out_dir = os.path.dirname(tf_file)
    expected_out = os.path.join(out_dir, '%s_%s_expected.tsv'%(date_str, social_var))
    expected = pd.DataFrame(expected, columns=[date_str])
    expected.to_csv(expected_out, sep='\t')
    diffusion_out = os.path.join(out_dir, '%s_%s_diffusion.tsv'%(date_str, social_var))
    print('writing to %s'%(diffusion_out))
    diffusion = pd.DataFrame(diffusion, columns=[date_str])
    diffusion.to_csv(diffusion_out, sep='\t')

if __name__ == '__main__':
    main()
