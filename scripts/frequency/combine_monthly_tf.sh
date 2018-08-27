# combine monthly frequency files
# COMBINED_NAME=2014_2015_tf
#COMBINED_NAME=2014_2016_tf
COMBINED_NAME=2013_2016_tf
# COMBINED_NAME=2015_2016_tf
DATA_DIR=../../data/frequency
# TF_FILES=($DATA_DIR/2014-06_tf.tsv $DATA_DIR/2014-07_tf.tsv $DATA_DIR/2014-08_tf.tsv $DATA_DIR/2014-09_tf.tsv $DATA_DIR/2014-10_tf.tsv $DATA_DIR/2014-11_tf.tsv $DATA_DIR/2014-12_tf.tsv $DATA_DIR/2015-01_tf.tsv $DATA_DIR/2015-02_tf.tsv $DATA_DIR/2015-03_tf.tsv $DATA_DIR/2015-04_tf.tsv $DATA_DIR/2015-05_tf.tsv)
TF_FILES=($DATA_DIR/2013-06_tf.tsv $DATA_DIR/2013-07_tf.tsv $DATA_DIR/2013-08_tf.tsv $DATA_DIR/2013-09_tf.tsv $DATA_DIR/2013-10_tf.tsv $DATA_DIR/2013-11_tf.tsv $DATA_DIR/2013-12_tf.tsv $DATA_DIR/2014-01_tf.tsv $DATA_DIR/2014-02_tf.tsv $DATA_DIR/2014-03_tf.tsv $DATA_DIR/2014-04_tf.tsv $DATA_DIR/2014-05_tf.tsv $DATA_DIR/2014-06_tf.tsv $DATA_DIR/2014-07_tf.tsv $DATA_DIR/2014-08_tf.tsv $DATA_DIR/2014-09_tf.tsv $DATA_DIR/2014-10_tf.tsv $DATA_DIR/2014-11_tf.tsv $DATA_DIR/2014-12_tf.tsv $DATA_DIR/2015-01_tf.tsv $DATA_DIR/2015-02_tf.tsv $DATA_DIR/2015-03_tf.tsv $DATA_DIR/2015-04_tf.tsv $DATA_DIR/2015-05_tf.tsv $DATA_DIR/2015-06_tf.tsv $DATA_DIR/2015-07_tf.tsv $DATA_DIR/2015-08_tf.tsv $DATA_DIR/2015-09_tf.tsv $DATA_DIR/2015-10_tf.tsv $DATA_DIR/2015-11_tf.tsv $DATA_DIR/2015-12_tf.tsv $DATA_DIR/2016-01_tf.tsv $DATA_DIR/2016-02_tf.tsv $DATA_DIR/2016-03_tf.tsv $DATA_DIR/2016-04_tf.tsv $DATA_DIR/2016-05_tf.tsv)
MIN_TF=0
# TF_FILES=($DATA_DIR/2014-06_tf.tsv $DATA_DIR/2014-07_tf.tsv $DATA_DIR/2014-08_tf.tsv $DATA_DIR/2014-09_tf.tsv $DATA_DIR/2014-10_tf.tsv $DATA_DIR/2014-11_tf.tsv $DATA_DIR/2014-12_tf.tsv $DATA_DIR/2015-01_tf.tsv $DATA_DIR/2015-02_tf.tsv $DATA_DIR/2015-03_tf.tsv $DATA_DIR/2015-04_tf.tsv $DATA_DIR/2015-05_tf.tsv $DATA_DIR/2015-06_tf.tsv $DATA_DIR/2015-07_tf.tsv $DATA_DIR/2015-08_tf.tsv $DATA_DIR/2015-09_tf.tsv $DATA_DIR/2015-10_tf.tsv $DATA_DIR/2015-11_tf.tsv $DATA_DIR/2015-12_tf.tsv $DATA_DIR/2016-01_tf.tsv $DATA_DIR/2016-02_tf.tsv $DATA_DIR/2016-03_tf.tsv $DATA_DIR/2016-04_tf.tsv $DATA_DIR/2016-05_tf.tsv)
OUTPUT=../../output/"$COMBINED_NAME"_output.txt
echo "${TF_FILES[@]}"
python combine_monthly_tf.py --combined_name $COMBINED_NAME --data_dir $DATA_DIR --tf_files "${TF_FILES[@]}" --min_tf $MIN_TF