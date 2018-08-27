"""
Fit a logistic distribution over all time series
and write the parameters to file.
"""
import pandas as pd
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_handler import get_default_vocab, smooth_stats
from argparse import ArgumentParser
import scipy.stats as st
from scipy.optimize import leastsq
from sklearn.preprocessing import MinMaxScaler
import os, re
from sklearn.metrics import r2_score

def residuals(p, y, x):
    loc, scale = p
    err = y - st.logistic.pdf(x, loc=loc, scale=scale)
    return err

def rescale(y, scaler):
    y_rescaled = scaler.fit_transform(y.values.reshape(-1, 1))
    y_rescaled /= y_rescaled.sum()
    y_rescaled = y_rescaled[:,0]
    return y_rescaled

def make_pdf(dist, params, x=None, size=10000):
    """Generate distribution's PDF"""
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    if(x is None):
        # Get same start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
        x = pd.np.linspace(start, end, size)
    # Build PDF and turn into pandas Series
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)
    return pdf

def main():
    parser = ArgumentParser()
    parser.add_argument('tf_file')
    parser.add_argument('stat_name')
    parser.add_argument('--out_dir', default='../../data/frequency/')
    args = parser.parse_args()
    tf_file = args.tf_file
    stat_name = args.stat_name
    out_dir = args.out_dir
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    # need to normalize stat to range [0,1) and norm to 1 because PROBABILITY
    scaler = MinMaxScaler()
    # vocab = get_default_vocab()
    # tf = pd.np.log10(smooth_stats(tf.loc[vocab, :].fillna(0, inplace=False)))
    tf_normed = tf.apply(lambda y: pd.Series(rescale(y, scaler), index=tf.columns), axis=1)
    idx = ['loc', 'scale']
    # initial parameters to optimize
    p0 = [1., 0.25]
    T = tf_normed.shape[1]
    X = pd.np.arange(T)
    logistic_params = tf_normed.apply(lambda y: pd.Series(leastsq(residuals, p0, args=(y, X))[0], index=idx), axis=1)
    # also compute R2 score
#     y_fit = logistic_params.apply(lambda r: pd.Series(st.logistic.pdf(X, loc=r[0], scale=r[1])), axis=1)
    y_fit = logistic_params.apply(lambda p: make_pdf(st.logistic, p.tolist(), x=X), axis=1)
    print('y fit has shape %s'%(str(y_fit.shape)))
    print('y fit has values \n%s'%(y_fit.head()))
    y_combined = pd.concat([tf_normed, y_fit], axis=1).fillna(tf_normed.min().min(), inplace=False)
    r2_scores = pd.DataFrame(y_combined.apply(lambda r: r2_score(r[:T], r[T:]), axis=1), columns=['R2'])
    logistic_params = pd.concat([logistic_params, r2_scores], axis=1)
    print('got final logistic params\n %s'%(logistic_params.head()))
    # write to file!!
    timeframe = re.findall('201[0-9]_201[0-9]', tf_file)[0]
    out_file = os.path.join(out_dir, '%s_%s_logistic_params.tsv'%(timeframe, stat_name))
    logistic_params.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()
