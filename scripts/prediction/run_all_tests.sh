# run all prediction tasks
# (1) relative importance
# (2) average dose response function
# (3) success/fail prediction
# (4) POS robustness check
# (5) survival analysis
OUT_DIR=../../output/results/
if [ ! -d $OUT_DIR ]; then
    mkdir $OUT_DIR
fi
Rscript relative_importance_tests.R $OUT_DIR # relative importance prediction
echo "finished relative importance"
Rscript get_average_dose_response_estimate_prob.R $OUT_DIR # ADRF prediction
Rscript plot_average_dose_response_function.R $OUT_DIR/ADRF_1_12/ # ADRF plotting
echo "finished causal inference"
python predict_success_k_month_window.py --out_dir $OUT_DIR # success prediction
python plot_success_k_month_window_lines.py --out_dir $OUT_DIR # plot success prediction
echo "finished prediction"
python predict_success_POS_tag.py --out_dir $OUT_DIR # success prediction POS robustness check
python ../frequency/plot_success_vs_failure_pos_DL_distribution.py --out_dir $OUT_DIR # within-category POS robustness check
echo "finished POS"
python ../frequency/plot_split_point_distribution_survivors.py --out_dir $OUT_DIR
python survival_analysis_tests.py --out_dir $OUT_DIR # survival analysis: coefficients, concordance
Rscript survival_analysis_tests.R $OUT_DIR # survival analysis: deviance
python plot_concordance_scores.py --out_dir $OUT_DIR # plot concordance scores
python compare_concordance_scores_between_models.py --out_dir $OUT_DIR
echo "finished survival analysis"