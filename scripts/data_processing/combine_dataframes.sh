# combine data split into
# python combine_dataframes.py ../data/frequency/word_social_frequency.tsv ../data/frequency/word_social_frequency_2015.tsv --out_file ../data/frequency/word_social_frequency.tsv --sort_cols word date
DATA_DIR=../../data/frequency
OUT_FILE=$DATA_DIR/2015_2016_tag_pcts.tsv
DATAFRAMES=$DATA_DIR/*tag_pcts.tsv
OUTPUT=../../output/combine_dataframes.txt
AXIS=1
(python combine_dataframes.py $DATAFRAMES --out_file $OUT_FILE --axis $AXIS > $OUTPUT)&
