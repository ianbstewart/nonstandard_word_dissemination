SOCIAL_VAR=thread
# SOCIAL_VAR=user
# SOCIAL_VAR=subreddit
TIMEFRAME=2013_2016
DATA_DIR=../../data/frequency
OUT_FILE=$DATA_DIR/"$TIMEFRAME"_unique_"$SOCIAL_VAR"_counts.tsv
COUNT_FILES=$DATA_DIR/201[0-9]-[01][0-9]_unique_"$SOCIAL_VAR"_counts.tsv
OUTPUT=../../output/"$TIMEFRAME"_unique_"$SOCIAL_VAR"_counts.txt
(python ../data_processing/combine_dataframes.py $COUNT_FILES --out_file $OUT_FILE --axis 1 > $OUTPUT)&
