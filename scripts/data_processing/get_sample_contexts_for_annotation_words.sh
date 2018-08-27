# get sample posts for all annotation words
WORD_LIST_DIR=../../data/frequency/word_lists
ANNOTATION_FILE=$WORD_LIST_DIR/top_200_success_scores.tsv
DATA_DIR=/hg190/corpora/reddit_comment_data/monthly_submission/2016/
SAMPLE_FILE=$DATA_DIR/RC_2016-05_clean.bz2
OUTPUT_DIR=$WORD_LIST_DIR/top_200_samples
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir $OUTPUT_DIR
fi
# get samples
N=200
# N=5
ANNOTATION_WORDS=$(cut -f1 $ANNOTATION_FILE | tail -n $N)
SAMPLE_COUNT=10
for ANNOTATION_WORD in $ANNOTATION_WORDS;
do
    echo $ANNOTATION_WORD
    SAMPLE_OUT_FILE=$OUTPUT_DIR/"$ANNOTATION_WORD"_samples.txt
    # build regex
    WORD_REGEX="^$ANNOTATION_WORD | $ANNOTATION_WORD | $ANNOTATION_WORD\$"
#     bzgrep "$ANNOTATION_WORD" $SAMPLE_FILE | head -n $SAMPLE_COUNT > $SAMPLE_OUT_FILE
    (bzgrep -P "$WORD_REGEX" $SAMPLE_FILE | head -n $SAMPLE_COUNT > $SAMPLE_OUT_FILE)&
done