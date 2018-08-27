# make bar plot of success/failure prediction accuracy
OUT_DIR=../../output
#ACCURACY_FILE=$OUT_DIR/bootstrap_matched_success_failure_k0_accuracy.tsv
# plotting multiple K
K_RANGE=(0)
# K_RANGE=(1 2 3 4 5 6 7)
for K in "${K_RANGE[@]}"; do
  ACCURACY_FILE=$OUT_DIR/bootstrap_matched_success_failure_k"$K"_non_differenced_accuracy_10fold.tsv
#   ACCURACY_FILE=$OUT_DIR/bootstrap_matched_success_failure_k"$K"_non_differenced_mean_accuracy_10fold.tsv
  OUTPUT_FILE=$OUT_DIR/plot_prediction_accuracy_k"$K".txt
  # (Rscript plot_prediction_accuracy.R $ACCURACY_FILE $OUT_DIR > $OUTPUT_FILE)&
  (python plot_prediction_accuracy.py $ACCURACY_FILE --out_dir $OUT_DIR > $OUTPUT_FILE)&
done