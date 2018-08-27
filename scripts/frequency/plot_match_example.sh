# plot example of matching words on frequency
DATA_DIR=../../data/frequency
# TF_FILE=$DATA_DIR/2013_2016_tf_norm.tsv
TF_FILE=$DATA_DIR/2013_2016_tf_norm_log.tsv
OUT_DIR=../../output
#MAIN_WORD='hatemail'
# MAIN_WORD='megaservers'
# SPLIT_POINT=13
# MATCH_WORDS=('untagged')
MAIN_WORD='fuckwit'
SPLIT_POINT=14
MATCH_WORDS=('fanart')
MATCH_K=(1)
OUTPUT=../../output/plot_match_example.txt
(python plot_match_example.py $MAIN_WORD $SPLIT_POINT $TF_FILE "${MATCH_WORDS[@]}" --match_k "${MATCH_K[@]}" --out_dir $OUT_DIR > $OUTPUT)&