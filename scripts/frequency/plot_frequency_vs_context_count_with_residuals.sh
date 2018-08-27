DATA_DIR=../../data/frequency
TF_FILE=$DATA_DIR/2013_2016_tf_norm_log.tsv
C3_FILE=$DATA_DIR/2013_2016_unique_3gram_counts.tsv
# EXAMPLE_INNOVATIONS=('someshit' 'prefab' 'yikes' 'aka')
EXAMPLE_INNOVATIONS=('yikes' 'aka')
OUT_DIR=../../output
(python plot_frequency_vs_context_count_with_residuals.py $TF_FILE $C3_FILE "${EXAMPLE_INNOVATIONS[@]}" $OUT_DIR)&