# compute tag estimates from tag percents
DATA_DIR=../../data/frequency
TAG_PCTS=$DATA_DIR/2013_2016_tag_pcts.tsv
OUT_FILE=$DATA_DIR/2013_2016_tag_estimates.tsv
(python get_tag_estimates.py $TAG_PCTS $OUT_FILE)&