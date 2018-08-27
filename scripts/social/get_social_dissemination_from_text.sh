# SOCIAL_VAR=user
# SOCIAL_VAR=subreddit
SOCIAL_VAR=thread
DATA_DIR=/net/hg190/corpora/reddit_comment_data/monthly_submission
CORPORA=($DATA_DIR/2014/RC_2014-07_normalized.bz2 $DATA_DIR/2014/RC_2014-10_normalized.bz2 $DATA_DIR/2014/RC_2014-11_normalized.bz2 $DATA_DIR/2014/RC_2014-12_normalized.bz2)
# CORPORA=$DATA_DIR/*/*clean_normalized.bz2
# CORPORA=($DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2)
# CORPORA=($DATA_DIR/2013/RC_2013-06_normalized.bz2 $DATA_DIR/2013/RC_2013-07_normalized.bz2 $DATA_DIR/2013/RC_2013-08_normalized.bz2 $DATA_DIR/2013/RC_2013-09_normalized.bz2 $DATA_DIR/2013/RC_2013-10_normalized.bz2 $DATA_DIR/2013/RC_2013-11_normalized.bz2 $DATA_DIR/2013/RC_2013-12_normalized.bz2 $DATA_DIR/2014/RC_2014-01_normalized.bz2 $DATA_DIR/2014/RC_2014-02_normalized.bz2 $DATA_DIR/2014/RC_2014-03_normalized.bz2 $DATA_DIR/2014/RC_2014-04_normalized.bz2 $DATA_DIR/2014/RC_2014-05_normalized.bz2)
# CORPORA=($DATA_DIR/2014/RC_2014-06_normalized.bz2 $DATA_DIR/2014/RC_2014-07_normalized.bz2 $DATA_DIR/2014/RC_2014-08_normalized.bz2 $DATA_DIR/2014/RC_2014-09_normalized.bz2 $DATA_DIR/2014/RC_2014-10_normalized.bz2 $DATA_DIR/2014/RC_2014-11_normalized.bz2 $DATA_DIR/2014/RC_2014-12_normalized.bz2 $DATA_DIR/2015/RC_2015-01_normalized.bz2 $DATA_DIR/2015/RC_2015-02_normalized.bz2 $DATA_DIR/2015/RC_2015-03_normalized.bz2 $DATA_DIR/2015/RC_2015-04_normalized.bz2 $DATA_DIR/2015/RC_2015-05_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-06_normalized.bz2 $DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2 $DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2)
# CORPORA=($DATA_DIR/2014/RC_2014-02_normalized.bz2 $DATA_DIR/2014/RC_2014-03_normalized.bz2 $DATA_DIR/2014/RC_2014-04_normalized.bz2 $DATA_DIR/2014/RC_2014-05_normalized.bz2 $DATA_DIR/2014/RC_2014-06_normalized.bz2 $DATA_DIR/2014/RC_2014-07_normalized.bz2 $DATA_DIR/2014/RC_2014-10_normalized.bz2 $DATA_DIR/2015/RC_2015-04_normalized.bz2 $DATA_DIR/2015/RC_2015-05_normalized.bz2 $DATA_DIR/2015/RC_2015-06_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2 $DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2 $DATA_DIR/2016/RC_2016-01_normalized.bz2 $DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-12_normalized.bz2 $DATA_DIR/2016/RC_2016-01_normalized.bz2 $DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
# CORPORA=($DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)

# FOR THREAD DIFFUSION: need batches of 4 to save memory
# CORPORA=($DATA_DIR/2015/RC_2015-06_normalized.bz2 $DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2 $DATA_DIR/2016/RC_2016-01_normalized.bz2)
# CORPORA=($DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-09_normalized.bz2 $DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
# for CORPUS in $CORPORA;
TF_FILE=../../data/frequency/2013_2016_tf_norm.tsv
OUT_DIR=../../output
# VOCAB=../../data/frequency/2013_2016_top_100000_vocab.tsv
for CORPUS in "${CORPORA[@]}";
do
    echo $CORPUS
    TIMEFRAME=$(echo $CORPUS | sed 's/.*\(201[0-9]-[01][0-9]\).*/\1/')
    OUTPUT=$OUT_DIR/"$TIMEFRAME"_"$SOCIAL_VAR"_diffusion_output.txt
    (python get_social_dissemination_from_text.py $CORPUS --social_var $SOCIAL_VAR --tf_file $TF_FILE > $OUTPUT)&
    # (python get_social_diffusion_from_text.py $CORPUS --social_var $SOCIAL_VAR --tf_file $TF_FILE --vocab $VOCAB > $OUTPUT)&
done
