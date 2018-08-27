# get all ngram counts
# DATA_DIR=/hg190/corpora/reddit_comment_data/monthly_submission
DATA_DIR=/net/hg190/corpora/reddit_comment_data/monthly_submission
# CORPORA=$DATA_DIR/*/*clean_normalized.bz2

# test sets
# CORPORA=($DATA_DIR/2015/RC_2015-07_clean.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-07_normalized.bz2)

# MEMORY OVERLOAD => sets of 4 at a time

# CORPORA=($DATA_DIR/2013/RC_2013-06_normalized.bz2 $DATA_DIR/2013/RC_2013-07_normalized.bz2 $DATA_DIR/2013/RC_2013-08_normalized.bz2 $DATA_DIR/2013/RC_2013-09_normalized.bz2 $DATA_DIR/2013/RC_2013-10_normalized.bz2 $DATA_DIR/2013/RC_2013-11_normalized.bz2 $DATA_DIR/2013/RC_2013-12_normalized.bz2 $DATA_DIR/2014/RC_2014-01_normalized.bz2)
# CORPORA=($DATA_DIR/2014/RC_2014-02_normalized.bz2 $DATA_DIR/2014/RC_2014-03_normalized.bz2 $DATA_DIR/2014/RC_2014-04_normalized.bz2 $DATA_DIR/2014/RC_2014-05_normalized.bz2 $DATA_DIR/2014/RC_2014-06_normalized.bz2 $DATA_DIR/2014/RC_2014-07_normalized.bz2 $DATA_DIR/2014/RC_2014-08_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-04_normalized.bz2 $DATA_DIR/2015/RC_2015-05_normalized.bz2 $DATA_DIR/2015/RC_2015-06_normalized.bz2 $DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2)
# CORPORA=($DATA_DIR/2014/RC_2014-10_normalized.bz2 $DATA_DIR/2014/RC_2014-11_normalized.bz2 $DATA_DIR/2014/RC_2014-12_normalized.bz2 $DATA_DIR/2015/RC_2015-01_normalized.bz2 $DATA_DIR/2015/RC_2015-02_normalized.bz2 $DATA_DIR/2015/RC_2015-03_normalized.bz2 $DATA_DIR/2015/RC_2015-04_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-05_normalized.bz2 $DATA_DIR/2015/RC_2015-06_normalized.bz2 $DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2 $DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2)
# CORPORA=($DATA_DIR/2016/RC_2016-01_normalized.bz2 $DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
# CORPORA=($DATA_DIR/2014/RC_2014-06_normalized.bz2 $DATA_DIR/2014/RC_2014-07_normalized.bz2 $DATA_DIR/2014/RC_2014-08_normalized.bz2 $DATA_DIR/2014/RC_2014-09_normalized.bz2)
# CORPORA=($DATA_DIR/2014/RC_2014-10_normalized.bz2 $DATA_DIR/2014/RC_2014-11_normalized.bz2 $DATA_DIR/2014/RC_2014-12_normalized.bz2 $DATA_DIR/2015/RC_2015-01_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-02_normalized.bz2 $DATA_DIR/2015/RC_2015-03_normalized.bz2 $DATA_DIR/2015/RC_2015-04_normalized.bz2 $DATA_DIR/2015/RC_2015-05_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-06_normalized.bz2 $DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2 $DATA_DIR/2016/RC_2016-01_normalized.bz2)
# CORPORA=($DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)

# full vocabulary => more memory
# CORPORA=($DATA_DIR/2013/RC_2013-06_clean.bz2 $DATA_DIR/2013/RC_2013-07_clean.bz2 $DATA_DIR/2013/RC_2013-08_clean.bz2 $DATA_DIR/2013/RC_2013-09_clean.bz2 $DATA_DIR/2013/RC_2013-10_clean.bz2 $DATA_DIR/2013/RC_2013-11_clean.bz2 $DATA_DIR/2013/RC_2013-12_clean.bz2 $DATA_DIR/2014/RC_2014-01_clean.bz2)
# CORPORA=($DATA_DIR/2014/RC_2014-02_clean.bz2 $DATA_DIR/2014/RC_2014-03_clean.bz2 $DATA_DIR/2014/RC_2014-04_clean.bz2 $DATA_DIR/2014/RC_2014-05_clean.bz2 $DATA_DIR/2014/RC_2014-06_clean.bz2 $DATA_DIR/2014/RC_2014-07_clean.bz2 $DATA_DIR/2014/RC_2014-08_clean.bz2 $DATA_DIR/2014/RC_2014-09_clean.bz2)
# CORPORA=($DATA_DIR/2014/RC_2014-10_clean.bz2 $DATA_DIR/2014/RC_2014-11_clean.bz2 $DATA_DIR/2014/RC_2014-12_clean.bz2 $DATA_DIR/2015/RC_2015-01_clean.bz2 $DATA_DIR/2015/RC_2015-02_clean.bz2 $DATA_DIR/2015/RC_2015-03_clean.bz2 $DATA_DIR/2015/RC_2015-04_clean.bz2 $DATA_DIR/2015/RC_2015-05_clean.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-06_clean.bz2 $DATA_DIR/2015/RC_2015-07_clean.bz2 $DATA_DIR/2015/RC_2015-08_clean.bz2 $DATA_DIR/2015/RC_2015-09_clean.bz2 $DATA_DIR/2015/RC_2015-10_clean.bz2 $DATA_DIR/2015/RC_2015-11_clean.bz2 $DATA_DIR/2015/RC_2015-12_clean.bz2 $DATA_DIR/2016/RC_2016-01_clean.bz2)
# CORPORA=($DATA_DIR/2016/RC_2016-02_clean.bz2 $DATA_DIR/2016/RC_2016-03_clean.bz2 $DATA_DIR/2016/RC_2016-04_clean.bz2 $DATA_DIR/2016/RC_2016-05_clean.bz2)
CORPORA=($DATA_DIR/2015/RC_2015-07_clean.bz2)
# CORPORA=($DATA_DIR/2015/RC_2015-02_clean.bz2 $DATA_DIR/2015/RC_2015-05_clean.bz2)

# N=2
N=3
FILE_SUFFIX=fullvocab
SAMPLE_PCT=100
for CORPUS in "${CORPORA[@]}";
do
    echo $CORPUS
    DATE=$(echo $CORPUS | sed 's/.*\(201[0-9]-[01][0-9]\).*/\1/')
    OUTPUT=../../output/"$DATE"_"$N"gram_counts.txt
#    (python get_ngram_counts.py --comment_files $CORPUS --n $N > $OUTPUT)&
    (python get_ngram_counts.py --comment_files $CORPUS --n $N --file_suffix $FILE_SUFFIX --sample_pct $SAMPLE_PCT > $OUTPUT)&
done
