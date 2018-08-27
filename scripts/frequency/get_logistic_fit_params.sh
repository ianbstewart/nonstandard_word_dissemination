# fit a logistic curve to all time series
DATA_DIR=../../data/frequency
TF_FILE=$DATA_DIR/2013_2016_tf_norm_log.tsv
STAT_NAME=tf_norm
TIMEFRAME=2013_2016
OUT_DIR=../../output
OUT_FILE=$OUT_DIR/"$TIMEFRAME"_"$STAT_NAME"_logistic_fit.txt
(python get_logistic_fit_params.py $TF_FILE $STAT_NAME --out_dir $DATA_DIR > $OUT_FILE)&