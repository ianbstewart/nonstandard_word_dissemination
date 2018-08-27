DATA_DIR=../../data/frequency
TF_FILE=$DATA_DIR/2013_2016_tf_norm_log.tsv
SOC_FILE=$DATA_DIR/2013_2016_unique_subreddit_counts.tsv
# EXAMPLE_INNOVATIONS=('someshit' 'prefab' 'yikes' 'aka')
EXAMPLE_INNOVATIONS=('probs' 'pls')
OUT_DIR=../../output
(python plot_frequency_vs_social_count_with_residuals.py $TF_FILE $SOC_FILE "${EXAMPLE_INNOVATIONS[@]}" $OUT_DIR)&