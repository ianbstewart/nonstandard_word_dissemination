# normalize pre-cleaned corpus
# external drive
# DATA_DIR="/Volumes/Seagate_Expansion_Drive/reddit_comments"
# conair
DATA_DIR=/hg190/corpora/reddit_comment_data/monthly_submission
OUTPUT_DIR=../../output/

# prebaked list of files to normalize
VOCAB=../../data/frequency/2013_2016_top_100000_vocab.tsv
CORPORA=($DATA_DIR/2013/RC_2013-06_clean.bz2 $DATA_DIR/2013/RC_2013-07_clean.bz2 $DATA_DIR/2013/RC_2013-08_clean.bz2 $DATA_DIR/2013/RC_2013-09_clean.bz2 $DATA_DIR/2013/RC_2013-10_clean.bz2 $DATA_DIR/2013/RC_2013-11_clean.bz2 $DATA_DIR/2013/RC_2013-12_clean.bz2 $DATA_DIR/2014/RC_2014-01_clean.bz2 $DATA_DIR/2014/RC_2014-02_clean.bz2 $DATA_DIR/2014/RC_2014-03_clean.bz2 $DATA_DIR/2014/RC_2014-04_clean.bz2 $DATA_DIR/2014/RC_2014-05_clean.bz2 $DATA_DIR/2014/RC_2014-06_clean.bz2 $DATA_DIR/2014/RC_2014-07_clean.bz2 $DATA_DIR/2014/RC_2014-08_clean.bz2 $DATA_DIR/2014/RC_2014-09_clean.bz2 $DATA_DIR/2014/RC_2014-10_clean.bz2 $DATA_DIR/2014/RC_2014-11_clean.bz2 $DATA_DIR/2014/RC_2014-12_clean.bz2 $DATA_DIR/2015/RC_2015-01_clean.bz2 $DATA_DIR/2015/RC_2015-02_clean.bz2 $DATA_DIR/2015/RC_2015-03_clean.bz2 $DATA_DIR/2015/RC_2015-04_clean.bz2 $DATA_DIR/2015/RC_2015-05_clean.bz2 $DATA_DIR/2015/RC_2015-06_clean.bz2 $DATA_DIR/2015/RC_2015-07_clean.bz2 $DATA_DIR/2015/RC_2015-08_clean.bz2 $DATA_DIR/2015/RC_2015-09_clean.bz2 $DATA_DIR/2015/RC_2015-10_clean.bz2 $DATA_DIR/2015/RC_2015-11_clean.bz2 $DATA_DIR/2015/RC_2015-12_clean.bz2 $DATA_DIR/2016/RC_2016-01_clean.bz2 $DATA_DIR/2016/RC_2016-02_clean.bz2 $DATA_DIR/2016/RC_2016-03_clean.bz2 $DATA_DIR/2016/RC_2016-04_clean.bz2 $DATA_DIR/2016/RC_2016-05_clean.bz2)
for CORPUS in "${CORPORA[@]}";
do
    echo $CORPUS
    OUTPUT=$(basename $CORPUS)
    OUTPUT=$OUTPUT_DIR/"$OUTPUT"_normalize_output.txt
    (python normalize_corpus.py $CORPUS --vocab $VOCAB > $OUTPUT)&
done