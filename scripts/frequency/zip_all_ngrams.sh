# zip ngrams to save space
DATA_DIR=../data/frequency
NGRAMS=$(ls $DATA_DIR/*gram_tf.tsv)
# for NGRAM in '${NGRAMS[@]}';
for NGRAM in $NGRAMS;
do
    ZIP_NAME=$NGRAM.zip
    zip $ZIP_NAME $NGRAM
    rm $NGRAM
done
