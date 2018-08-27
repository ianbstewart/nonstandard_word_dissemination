# plot C3 distribution across POS groups for growth/growth-decline words
DATA_DIR=../../data/frequency
OUT_DIR=../../output
MATCH_STAT=$DATA_DIR/2013_2016_tf_norm_log.tsv
PLOT_STAT=$DATA_DIR/2013_2016_3gram_residuals.tsv
TAG_PCTS=$DATA_DIR/2013_2016_tag_pcts.tsv
OUTPUT=$OUT_DIR/plot_success_vs_failure_pos_DL_distribution.txt
(python plot_success_vs_failure_pos_DL_distribution.py $DATA_DIR $MATCH_STAT $PLOT_STAT $TAG_PCTS --out_dir $OUT_DIR > $OUTPUT)&