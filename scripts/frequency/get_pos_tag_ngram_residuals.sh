# compute residuals between frequency and unique POS tag ngram counts
DATA_DIR=../../data/frequency
OUT_DIR=../../output
# TF_FILE=$DATA_DIR/2013_2016_tf_norm.tsv
TF_FILE=$DATA_DIR/2013_2016_tf.tsv
UP3_FILE=$DATA_DIR/2013_2016_unique_3gram_tag_counts.tsv
OUTPUT=$OUT_DIR/2013_2016_pos_tag_ngram_residuals.txt
# (python get_pos_tag_ngram_residuals.py $TF_FILE $UP3_FILE --out_dir $DATA_DIR > $OUTPUT)&
(python get_pos_tag_ngram_residuals.py $TF_FILE $UP3_FILE --out_dir $DATA_DIR)&