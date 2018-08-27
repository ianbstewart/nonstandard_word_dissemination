# compute average prediction accuracy
OUT_DIR=../../output
K_VALS=(1 2 3 4 5 6)
for K in "${K_VALS[@]}";
do
    ACCURACY_FILE=$OUT_DIR/bootstrap_matched_success_failure_k"$K"_non_differenced_accuracy.tsv
    python average_prediction_accuracy.py $ACCURACY_FILE
    ACCURACY_AVERAGE_FILE=${ACCURACY_FILE/.tsv/_average.tsv}
    # print scores
    echo $K
    cat $ACCURACY_AVERAGE_FILE
done