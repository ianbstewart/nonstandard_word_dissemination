# get sample of word queried
Q_WORD=$1
SAMPLE_SIZE=50
OUT_DIR=../../output
# DATE=2016-05
YEAR=2015
MONTH=06
DATE=$YEAR-$MONTH
OUTPUT=$OUT_DIR/"$DATE"_"$Q_WORD"_sample
DATA_DIR=/hg190/corpora/reddit_comment_data/monthly_submission/$YEAR
# CORPUS=$DATA_DIR/RC_"$DATE"_normalized.bz2
CORPUS=$DATA_DIR/RC_"$DATE"_clean.bz2
(bzgrep " $Q_WORD " $CORPUS | head -$SAMPLE_SIZE > $OUTPUT)&
