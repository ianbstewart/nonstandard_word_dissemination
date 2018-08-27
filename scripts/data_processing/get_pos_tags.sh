# test tweet tagger on Reddit data to remove some of those named entities
DATA_DIR=/hg190/corpora/reddit_comment_data/monthly_submission
# YEAR=2015
# DATE=RC_2015-06
# CORPUS=$DATA_DIR/"$DATE"_normalized.bz2
CORPORA=($DATA_DIR/2014/RC_2014-01_normalized.bz2 $DATA_DIR/2014/RC_2014-02_normalized.bz2 $DATA_DIR/2014/RC_2014-03_normalized.bz2 $DATA_DIR/2014/RC_2014-04_normalized.bz2 $DATA_DIR/2014/RC_2014-05_normalized.bz2 $DATA_DIR/2014/RC_2014-06_normalized.bz2 $DATA_DIR/2014/RC_2014-07_normalized.bz2 $DATA_DIR/2014/RC_2014-08_normalized.bz2 $DATA_DIR/2014/RC_2014-09_normalized.bz2 $DATA_DIR/2014/RC_2014-10_normalized.bz2 $DATA_DIR/2014/RC_2014-11_normalized.bz2 $DATA_DIR/2014/RC_2014-12_normalized.bz2 $DATA_DIR/2015/RC_2015-01_normalized.bz2 $DATA_DIR/2015/RC_2015-02_normalized.bz2 $DATA_DIR/2015/RC_2015-03_normalized.bz2 $DATA_DIR/2015/RC_2015-04_normalized.bz2 $DATA_DIR/2015/RC_2015-05_normalized.bz2 $DATA_DIR/2015/RC_2015-06_normalized.bz2 $DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2 $DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2 $DATA_DIR/2016/RC_2016-01_normalized.bz2 $DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
# CORPORA=($DATA_DIR/2013/RC_2013-06_normalized.bz2 $DATA_DIR/2013/RC_2013-07_normalized.bz2 $DATA_DIR/2013/RC_2013-08_normalized.bz2 $DATA_DIR/2013/RC_2013-09_normalized.bz2 $DATA_DIR/2013/RC_2013-10_normalized.bz2 $DATA_DIR/2013/RC_2013-11_normalized.bz2 $DATA_DIR/2013/RC_2013-12_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2 $DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2 $DATA_DIR/2016/RC_2016-01_normalized.bz2 $DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
for CORPUS in "${CORPORA[@]}"; do
    TIMEFRAME=$(echo $CORPUS | sed 's/.*\(201[0-9]-[01][0-9]\).*/\1/')
    DATA_DIR=$(dirname "$CORPUS")
    TXT=$DATA_DIR/"$TIMEFRAME"_normalized.txt
    OUT_DIR=../../output
    OUTPUT=$OUT_DIR/"$TIMEFRAME"_tags.txt
    (if [ ! -f $TXT ]; then bzcat $CORPUS > $TXT; fi && bash ark-tweet-nlp-0.3.2/runTagger.sh --output-format conll $TXT > $OUTPUT && bzip2 $OUTPUT)&
done