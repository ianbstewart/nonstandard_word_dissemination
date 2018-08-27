# generate match examples
DATA_DIR=../../data/frequency
MATCH_STAT=$DATA_DIR/2013_2016_tf_norm_log.tsv
K=0
OUT_DIR=../../output
(Rscript get_match_examples.R $MATCH_STAT $K $OUT_DIR $DATA_DIR)&