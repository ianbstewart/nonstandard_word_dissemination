# SOCIAL_VAR=thread
SOCIAL_VAR=user
# SOCIAL_VAR=subreddit
TIMEFRAME=2013_2016
DATA_DIR=../../data/frequency
OUT_FILE=$DATA_DIR/"$TIMEFRAME"_"$SOCIAL_VAR"_diffusion.tsv
COUNT_FILES=$DATA_DIR/201[0-9]-[01][0-9]_"$SOCIAL_VAR"_diffusion.tsv
OUTPUT=../../output/"$TIMEFRAME"_combine_"$SOCIAL_VAR"_diffusion.txt
(python ../data_processing/combine_dataframes.py $COUNT_FILES $OUT_FILE --axis 1 > $OUTPUT)&
