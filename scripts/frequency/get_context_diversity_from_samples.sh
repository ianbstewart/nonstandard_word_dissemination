# compute context diversity for provided words
# based on unique number of contexts
# WORD_FILE=../../data/frequency/word_lists/2013_2016_growth_words_clean_test.csv
WORD_FILE=../../data/frequency/word_lists/2013_2016_growth_words_clean.csv
N=3
NGRAM_FILES=(../../data/frequency/2015-01_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-02_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-03_"$N"gram_tf_fullvocab.tsv../../data/frequency/2015-04_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-05_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-06_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-07_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-08_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-09_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-10_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-11_"$N"gram_tf_fullvocab.tsv ../../data/frequency/2015-12_"$N"gram_tf_fullvocab.tsv)
# TIMEFRAME=2015-01
# NGRAM_FILE=../../data/frequency/"$TIMEFRAME"_"$N"gram_tf_fullvocab.tsv
for NGRAM_FILE in "${NGRAM_FILES[@]}";
do
    TIMEFRAME=$(echo $NGRAM_FILE | sed -i 's/.*\(201[0-6]-[0-1][0-9]\).*/\1/')
    echo $TIMEFRAME
    OUTPUT=../../output/"$TIMEFRAME"_context_diversity_sampling.txt
    OUT_FILE=../../data/frequency/"$TIMEFRAME"_"$N"gram_context_samples.tsv
    python get_context_diversity_from_samples.py $WORD_FILE $NGRAM_FILE $OUT_FILE > $OUTPUT
    # (python get_context_diversity_from_samples.py $WORD_FILE $NGRAM_FILE $OUT_FILE > $OUTPUT)&
done
