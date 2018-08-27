"""
Compute 2-part piecewise fit over all frequency time series
using discrete optimization: try every possible split point,
compute the R2, and choose the split point with the best R2.
CLUNKY BUT IT WORKS.
"""
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import r2_score
from argparse import ArgumentParser
import os

class DummyCounter:
    def __init__(self):
        self.ctr = 0
def optimize_split_point(X, y, CTR=None, split_points=None):
    """
    Optimize split point t_hat in X time series s.t. 
    linear fit to y[:t_hat] and y[t_hat:] maximizes R2 score.

    Parameters:
    -----------
    X : array-like
    y : array-like
    CTR : DummyCounter
    split_points : array-like
    Integer values of allowed split points.

    Returns:
    --------
    t_hat : int
    m1_hat : float
    m2_hat : float
    b_hat : float
    r2_hat : float
    """
    if(split_points is None):
        split_points = X[2:-2]
    # compute R2 for each split point
    y_mean = y.mean()
    ss_tot = ((y - y_mean)**2).sum()
    r2_hat = -1.
    m1_hat = 0.
    m2_hat = 0.
    b_hat = 0.
    t_hat = 0.
    idx = ['s', 'm1', 'm2', 'R2']
    for t in split_points:
        X1 = X[:t]
        y1 = y[:t]
        m1, b1, r1, p1, err1 = linregress(X1, y1)
        # solve for m2 using b2, x
        b2 = m1*X[t] + b1
        X2 = X[t+1:]
        y2 = y[t+1:] - b2
        m2, _, r2, p2, err2 = linregress(X2, y2)
        y1_pred = m1*X[:t] + b1
        y2_pred = m2*X[t:] + b2
        y_pred = pd.np.concatenate([y1_pred, y2_pred])
        ss_res = ((y - y_pred)**2).sum()
        r2_s = 1 - ss_res / ss_tot
        if(r2_s > r2_hat):
            r2_hat = r2_s
            t_hat = t
            m1_hat = m1
            m2_hat = m2
            b_hat = b1
    CTR.ctr += 1
    if(CTR.ctr % 1000 == 0):
        print('processed %d words'%(CTR.ctr))
    return t_hat, m1_hat, m2_hat, b_hat, r2_hat
  
def main():
    parser = ArgumentParser()
    parser.add_argument('tf_file')
    args = parser.parse_args()
    tf_file = args.tf_file
    out_dir = os.path.dirname(tf_file)
    out_file = os.path.join(out_dir, '2013_2016_tf_norm_log_2_piecewise_discrete.tsv')

    ## load data
    tf = pd.read_csv(tf_file, sep='\t', index_col=0)
    tf.fillna(tf.min().min(), inplace=True)
    X = pd.np.arange(tf.shape[1])
    split_points = X[1:-1]

    ## optimize split points
    idx = ['t', 'm1', 'm2', 'b', 'R2']
    CTR = DummyCounter()
    split_point_params = tf.apply(lambda y: pd.Series(optimize_split_point(X, y, CTR=CTR)), axis=1)
    split_point_params.columns = idx
    print('done')
    # write to file!!
    split_point_params.to_csv(out_file, sep='\t')

if __name__ == '__main__':
    main()