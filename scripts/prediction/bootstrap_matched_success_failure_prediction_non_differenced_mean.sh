# bootstrap matching and do prediction for success/failure using mean covariate values
# write accuracy and coefficients to file
DATA_DIR=../../data/frequency
OUT_DIR=../../output
BOOTSTRAP_ITER=100
KMIN=1
KMAX=7
for K in `seq $KMIN $KMAX`;
do
    echo $K
    OUTPUT_FILE=$OUT_DIR/bootstrap_matched_success_failure_prediction_k"$K"_non_differenced_mean.txt
    (Rscript bootstrap_matched_success_failure_prediction_non_differenced_mean.R $DATA_DIR $OUT_DIR $BOOTSTRAP_ITER $K > $OUTPUT_FILE)&
done