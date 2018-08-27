URL_BASE=https://archive.org/download/2015_reddit_comments_corpus/reddit_data
YEAR=2014
YEAR=2013
URL_BASE=$URL_BASE/$YEAR
# MONTHS=(07 08 09 10 11 12)
# MONTHS=(01 02 03 04 05)
MONTHS=(01 02 03 04 05 06 07 08 09 10 11 12)
OUT_DIR=/hg190/corpora/reddit_comment_data/monthly_submission/$YEAR
if [ ! -d "$OUT_DIR" ]; then
    mkdir $OUT_DIR
fi
for MONTH in "${MONTHS[@]}";
do
    FULL_URL=$URL_BASE/RC_"$YEAR"-$MONTH.bz2
    wget $FULL_URL -P $OUT_DIR/
done