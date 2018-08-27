# plot survivor curve
OUT_DIR=../../output
OUTPUT=$OUT_DIR/plot_split_point_distribution_survivors.txt
(python plot_split_point_distribution_survivors.py --out_dir $OUT_DIR > $OUTPUT)&