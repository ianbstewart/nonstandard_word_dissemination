# plot examples of growth-decline words with best fit lines
DATA_DIR=../../data/frequency
# PIECEWISE_WORD="wot"
PIECEWISE_WORD="sorta"
LOGISTIC_WORD="iifym"
TF_FILE=$DATA_DIR/2013_2016_tf_norm_log.tsv
PIECEWISE_PARAMS=$DATA_DIR/2013_2016_tf_norm_2_piecewise.tsv
LOGISTIC_PARAMS=$DATA_DIR/2013_2016_tf_norm_logistic_params.tsv
# OUT_DIR=../../data/images/
OUT_DIR=../../output/
(python plot_failure_examples.py --piecewise_word $PIECEWISE_WORD --logistic_word $LOGISTIC_WORD --tf_file $TF_FILE --piecewise_params $PIECEWISE_PARAMS --logistic_params $LOGISTIC_PARAMS --out_dir $OUT_DIR)&