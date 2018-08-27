# get 2-part piecewise parameters over all frequency time series
DATA_DIR=../../data/frequency
TF=$DATA_DIR/2013_2016_tf_norm_log.tsv
OUT_DIR=../../output
OUTPUT=$OUT_DIR/get_piecewise_fit_params_discrete.txt
(python get_piecewise_fit_params_discrete.py $TF > $OUTPUT)&