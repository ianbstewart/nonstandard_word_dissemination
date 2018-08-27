DATA_DIR=../../output
# TAG_FILES=($DATA_DIR/2015-06_tags.txt.bz2 $DATA_DIR/2015-07_tags.txt.bz2 $DATA_DIR/2015-08_tags.txt.bz2 $DATA_DIR/2015-09_tags.txt.bz2 $DATA_DIR/2015-10_tags.txt.bz2 $DATA_DIR/2015-11_tags.txt.bz2 $DATA_DIR/2015-12_tags.txt.bz2 $DATA_DIR/2016-01_tags.txt.bz2 $DATA_DIR/2016-02_tags.txt.bz2 $DATA_DIR/2016-03_tags.txt.bz2 $DATA_DIR/2016-04_tags.txt.bz2 $DATA_DIR/2016-05_tags.txt.bz2)
TAG_FILES=($DATA_DIR/2013-06_tags.txt.bz2 $DATA_DIR/2013-07_tags.txt.bz2 $DATA_DIR/2013-08_tags.txt.bz2 $DATA_DIR/2013-09_tags.txt.bz2 $DATA_DIR/2013-10_tags.txt.bz2 $DATA_DIR/2013-11_tags.txt.bz2 $DATA_DIR/2013-12_tags.txt.bz2 $DATA_DIR/2014-01_tags.txt.bz2 $DATA_DIR/2014-02_tags.txt.bz2 $DATA_DIR/2014-03_tags.txt.bz2 $DATA_DIR/2014-04_tags.txt.bz2 $DATA_DIR/2014-05_tags.txt.bz2 $DATA_DIR/2014-06_tags.txt.bz2 $DATA_DIR/2014-07_tags.txt.bz2 $DATA_DIR/2014-08_tags.txt.bz2 $DATA_DIR/2014-09_tags.txt.bz2 $DATA_DIR/2014-10_tags.txt.bz2 $DATA_DIR/2014-11_tags.txt.bz2 $DATA_DIR/2014-12_tags.txt.bz2 $DATA_DIR/2015-01_tags.txt.bz2 $DATA_DIR/2015-02_tags.txt.bz2 $DATA_DIR/2015-03_tags.txt.bz2 $DATA_DIR/2015-04_tags.txt.bz2 $DATA_DIR/2015-05_tags.txt.bz2 $DATA_DIR/2015-06_tags.txt.bz2 $DATA_DIR/2015-07_tags.txt.bz2 $DATA_DIR/2015-08_tags.txt.bz2 $DATA_DIR/2015-09_tags.txt.bz2 $DATA_DIR/2015-10_tags.txt.bz2 $DATA_DIR/2015-11_tags.txt.bz2 $DATA_DIR/2015-12_tags.txt.bz2 $DATA_DIR/2016-01_tags.txt.bz2 $DATA_DIR/2016-02_tags.txt.bz2 $DATA_DIR/2016-03_tags.txt.bz2 $DATA_DIR/2016-04_tags.txt.bz2 $DATA_DIR/2016-05_tags.txt.bz2)
N=2
# N=3
OUT_DIR=../../output
for TAG_FILE in "${TAG_FILES[@]}";
do
    TIMEFRAME=$(echo $TAG_FILE | sed 's/.*\(201[0-9]-[01][0-9]\).*/\1/')
    OUTPUT=$OUT_DIR/"$TIMEFRAME"_pos_tag_ngram_context_counts.txt
    (python get_pos_tag_ngram_context_counts.py $TAG_FILE --n $N > $OUTPUT)&
done