# significance testing between feature sets' concordance scores
OUT_DIR=../../output
CONCORDANCE_SCORES=$OUT_DIR/cox_regression_concordance_10_fold_scores.tsv
(python compare_concordance_scores_between_models.py $CONCORDANCE_SCORES --out_dir $OUT_DIR)&