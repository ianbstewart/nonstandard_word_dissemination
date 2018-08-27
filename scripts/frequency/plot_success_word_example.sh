# plot examples of growth word with best fit line
DATA_DIR=../../data/frequency
SUCCESS_WORD="kinda"
TF_FILE=$DATA_DIR/2013_2016_tf_norm_log.tsv
OUT_DIR=../../output/
(python plot_success_word_example.py --success_word $SUCCESS_WORD --tf_file $TF_FILE --out_dir $OUT_DIR)&