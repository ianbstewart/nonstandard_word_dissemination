# compute average over all cells in dataframes
DATA_DIR=../../data/frequency
DATA_NAME=tag_pcts
DATAFRAMES=$DATA_DIR/*"$DATA_NAME".tsv
TIMEFRAME=2015_2016
OUTPUT=../../output/average_dataframes.txt
(python average_dataframes.py $DATAFRAMES --timeframe $TIMEFRAME > $OUTPUT)&