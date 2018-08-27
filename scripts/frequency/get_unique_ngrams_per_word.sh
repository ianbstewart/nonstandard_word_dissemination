# count unique ngrams for all words in vocab for all timesteps
# N=2
N=3
NGRAM_POS=0
# NGRAM_POS=1
# NGRAM_POS=2
DATA_DIR=../../data/frequency
# NGRAM_FILES=($DATA_DIR/2013-06_"$N"gram_tf.tsv $DATA_DIR/2013-07_"$N"gram_tf.tsv $DATA_DIR/2013-08_"$N"gram_tf.tsv $DATA_DIR/2013-09_"$N"gram_tf.tsv $DATA_DIR/2013-10_"$N"gram_tf.tsv $DATA_DIR/2013-11_"$N"gram_tf.tsv $DATA_DIR/2013-12_"$N"gram_tf.tsv $DATA_DIR/2014-01_"$N"gram_tf.tsv $DATA_DIR/2014-02_"$N"gram_tf.tsv $DATA_DIR/2014-03_"$N"gram_tf.tsv $DATA_DIR/2014-04_"$N"gram_tf.tsv $DATA_DIR/2014-05_"$N"gram_tf.tsv $DATA_DIR/2014-06_"$N"gram_tf.tsv $DATA_DIR/2014-07_"$N"gram_tf.tsv $DATA_DIR/2014-08_"$N"gram_tf.tsv $DATA_DIR/2014-09_"$N"gram_tf.tsv $DATA_DIR/2014-10_"$N"gram_tf.tsv $DATA_DIR/2014-11_"$N"gram_tf.tsv $DATA_DIR/2014-12_"$N"gram_tf.tsv $DATA_DIR/2015-01_"$N"gram_tf.tsv $DATA_DIR/2015-02_"$N"gram_tf.tsv $DATA_DIR/2015-03_"$N"gram_tf.tsv $DATA_DIR/2015-04_"$N"gram_tf.tsv $DATA_DIR/2015-05_"$N"gram_tf.tsv $DATA_DIR/2015-06_"$N"gram_tf.tsv $DATA_DIR/2015-07_"$N"gram_tf.tsv $DATA_DIR/2015-08_"$N"gram_tf.tsv $DATA_DIR/2015-09_"$N"gram_tf.tsv $DATA_DIR/2015-10_"$N"gram_tf.tsv $DATA_DIR/2015-11_"$N"gram_tf.tsv $DATA_DIR/2015-12_"$N"gram_tf.tsv $DATA_DIR/2016-01_"$N"gram_tf.tsv $DATA_DIR/2016-02_"$N"gram_tf.tsv $DATA_DIR/2016-03_"$N"gram_tf.tsv $DATA_DIR/2016-04_"$N"gram_tf.tsv $DATA_DIR/2016-05_"$N"gram_tf.tsv)
NGRAM_FILES=($DATA_DIR/2013-06_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2013-07_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2013-08_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2013-09_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2013-10_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2013-11_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2013-12_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-01_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-02_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-03_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-04_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-05_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-06_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-07_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-08_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-09_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-10_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-11_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2014-12_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-01_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-02_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-03_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-04_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-05_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-06_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-07_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-08_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-09_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-10_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-11_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-12_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2016-01_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2016-02_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2016-03_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2016-04_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2016-05_"$N"gram_tf_fullvocab.tsv)
# NGRAM_FILES=($DATA_DIR/2015-07_"$N"gram_tf_fullvocab.tsv)
# NGRAM_FILES=($DATA_DIR/2015-02_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-03_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-04_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-05_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-06_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2015-08_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2016-02_"$N"gram_tf_fullvocab.tsv $DATA_DIR/2016-05_"$N"gram_tf_fullvocab.tsv)
TIMEFRAME=2013_2016
FILE_SUFFIX=fullvocab

# one corpus at a time
# TIMEFRAME=2014-05
# NGRAM_FILES=($DATA_DIR/"$TIMEFRAME"_"$N"gram_tf_fullvocab.tsv)

# TF_FILE=../../data/frequency/"$TIMEFRAME"_tf.tsv
# OUTPUT=../../output/"$TIMEFRAME"_unique_"$N"grams_"$NGRAM_POS"pos_per_word.txt
for NGRAM_FILE in "${NGRAM_FILES[@]}";
do
    TIMEFRAME=$(echo $NGRAM_FILE | sed 's/.*\(201[0-9]-[01][0-9]\).*/\1/')
    # echo $TIMEFRAME
    OUTPUT=../../output/"$TIMEFRAME"_unique_"$N"grams_"$NGRAM_POS"pos_per_word.txt
    (python get_unique_ngrams_per_word.py $NGRAM_FILE --n $N --timeframe $TIMEFRAME --ngram_pos $NGRAM_POS --file_suffix $FILE_SUFFIX > $OUTPUT)&
done