# get ngram residuals from linear regression on log-f vs. log-ngram count
DATA_DIR=../../data/frequency
# TF=$DATA_DIR/2015_2016_tf.tsv
TIMEFRAME=2013_2016
TF=$DATA_DIR/"$TIMEFRAME"_tf.tsv
N=2
# N=3
NGRAM=$DATA_DIR/"$TIMEFRAME"_unique_"$N"gram_counts.tsv
OUTPUT=../../output/"$N"gram_residuals.txt
(python get_ngram_residuals.py --tf $TF --ngram $NGRAM > $OUTPUT)&