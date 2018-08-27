# predict success/failure on bootstrapped sample of word pairs,
# write accuracy and coefficients to file
DATA_DIR=../../data/frequency
OUT_DIR=../../output
BOOTSTRAP_ITER=100
K=0
# K=1
# K=2
# K=3
# K=4
# K=5
# K=6
for K in `seq 1 6`;
do
    echo $K
    OUTPUT_FILE=$OUT_DIR/bootstrap_matched_success_failure_prediction_k"$K"_non_differenced.txt
    (Rscript bootstrap_matched_success_failure_prediction_non_differenced.R $DATA_DIR $OUT_DIR $BOOTSTRAP_ITER $K > $OUTPUT_FILE)&
done