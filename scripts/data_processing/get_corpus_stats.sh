DATA_DIR=/net/hg190/corpora/reddit_comment_data/monthly_submission
CORPORA=($DATA_DIR/2013/RC_2013-06_normalized.bz2 $DATA_DIR/2013/RC_2013-07_normalized.bz2 $DATA_DIR/2013/RC_2013-08_normalized.bz2 $DATA_DIR/2013/RC_2013-09_normalized.bz2 $DATA_DIR/2013/RC_2013-10_normalized.bz2 $DATA_DIR/2013/RC_2013-11_normalized.bz2 $DATA_DIR/2013/RC_2013-12_normalized.bz2 $DATA_DIR/2014/RC_2014-01_normalized.bz2 $DATA_DIR/2014/RC_2014-02_normalized.bz2 $DATA_DIR/2014/RC_2014-03_normalized.bz2 $DATA_DIR/2014/RC_2014-04_normalized.bz2 $DATA_DIR/2014/RC_2014-05_normalized.bz2 $DATA_DIR/2014/RC_2014-06_normalized.bz2 $DATA_DIR/2014/RC_2014-07_normalized.bz2 $DATA_DIR/2014/RC_2014-08_normalized.bz2 $DATA_DIR/2014/RC_2014-09_normalized.bz2 $DATA_DIR/2014/RC_2014-10_normalized.bz2 $DATA_DIR/2014/RC_2014-11_normalized.bz2 $DATA_DIR/2014/RC_2014-12_normalized.bz2 $DATA_DIR/2015/RC_2015-01_normalized.bz2 $DATA_DIR/2015/RC_2015-02_normalized.bz2 $DATA_DIR/2015/RC_2015-03_normalized.bz2 $DATA_DIR/2015/RC_2015-04_normalized.bz2 $DATA_DIR/2015/RC_2015-05_normalized.bz2 $DATA_DIR/2015/RC_2015-06_normalized.bz2 $DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2 $DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2 $DATA_DIR/2016/RC_2016-01_normalized.bz2 $DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-12_normalized.bz2)
OUT_DIR=../../data/metadata/
OUTPUT_DIR=../../output
TIMEFRAME=2013_2016
OUTPUT=$OUTPUT_DIR/"$TIMEFRAME"_corpus_stats.txt
(python get_corpus_stats.py "${CORPORA[@]}" --out_dir $OUT_DIR > $OUTPUT)&
# for CORPUS in "${CORPORA[@]}";
# do
#     TIMEFRAME=$(echo $CORPUS | sed 's/.*\(201[0-9]-[01][0-9]\).*/\1/')
#     OUTPUT=$OUTPUT_DIR/"$TIMEFRAME"_corpus_stats.txt
#     (python get_corpus_stats.py $CORPUS --out_dir $OUT_DIR > $OUTPUT)&
# done