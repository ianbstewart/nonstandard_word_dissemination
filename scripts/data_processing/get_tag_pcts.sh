DATA_DIR=../../output
# TAG_FILES=$DATA_DIR/*tags.txt
TAG_FILES=($DATA_DIR/2014-02_tags.txt)
# TAG_FILES=($DATA_DIR/2014-01_tags.txt $DATA_DIR/2014-02_tags.txt $DATA_DIR/2014-03_tags.txt $DATA_DIR/2014-04_tags.txt $DATA_DIR/2014-05_tags.txt $DATA_DIR/2014-06_tags.txt $DATA_DIR/2014-07_tags.txt $DATA_DIR/2014-08_tags.txt $DATA_DIR/2014-09_tags.txt $DATA_DIR/2014-10_tags.txt $DATA_DIR/2014-11_tags.txt $DATA_DIR/2014-12_tags.txt $DATA_DIR/2015-01_tags.txt $DATA_DIR/2015-02_tags.txt $DATA_DIR/2015-03_tags.txt $DATA_DIR/2015-04_tags.txt $DATA_DIR/2015-05_tags.txt $DATA_DIR/2015-06_tags.txt $DATA_DIR/2015-07_tags.txt $DATA_DIR/2015-08_tags.txt $DATA_DIR/2015-09_tags.txt $DATA_DIR/2015-10_tags.txt $DATA_DIR/2015-11_tags.txt $DATA_DIR/2015-12_tags.txt $DATA_DIR/2016-01_tags.txt $DATA_DIR/2016-02_tags.txt $DATA_DIR/2016-03_tags.txt $DATA_DIR/2016-04_tags.txt $DATA_DIR/2016-05_tags.txt)
for TAG_FILE in "${TAG_FILES[@]}"; 
do
    (python get_tag_pcts.py $TAG_FILE)&
done