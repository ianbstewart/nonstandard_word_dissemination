# combine ngram counts with different positions
DATA_DIR=../../data/frequency
# N=2
N=3
TIMEFRAME=2013_2016
# NGRAM_FILES=($DATA_DIR/"$TIMEFRAME"_"$N"gram_0pos_counts.tsv $DATA_DIR/"$TIMEFRAME"_"$N"gram_1pos_counts.tsv)
NGRAM_FILES=($DATA_DIR/"$TIMEFRAME"_"$N"gram_0pos_counts.tsv $DATA_DIR/"$TIMEFRAME"_"$N"gram_1pos_counts.tsv $DATA_DIR/"$TIMEFRAME"_"$N"gram_2pos_counts.tsv)
OUT_FILE=$DATA_DIR/"$TIMEFRAME"_unique_"$N"gram_counts.tsv
OUTPUT=../../output/"$TIMEFRAME"_unique_"$N"gram_counts_combined.txt
(python ../data_processing/add_dataframes.py "${NGRAM_FILES[@]}" $OUT_FILE > $OUTPUT)&