# test difference in DC values between POS groups
DATA_DIR=../../data/frequency
TAG_ESTIMATES=$DATA_DIR/2013_2016_tag_estimates.tsv
DEPENDENT_VAR=$DATA_DIR/2013_2016_3gram_residuals.tsv
DEPENDENT_VAR_NAME=DC
OUT_DIR=../../output
TAGS_TO_TEST=(! A N R V ~)
# (python anova_pos_DC_test.py $TAG_ESTIMATES $DEPENDENT_VAR $DEPENDENT_VAR_NAME $OUT_DIR --tags_to_test "${TAGS_TO_TEST[@]}")&
(python anova_pos_DL_test.py $TAG_ESTIMATES $DEPENDENT_VAR $DEPENDENT_VAR_NAME $OUT_DIR)&