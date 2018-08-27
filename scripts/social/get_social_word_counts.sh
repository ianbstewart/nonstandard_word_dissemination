# compute social counts over all comment files
#DATA_DIR=/mnt/new_hg190/corpora/reddit_comment_data/monthly_submission
DATA_DIR=/hg190/corpora/reddit_comment_data/monthly_submission
# YEAR=2015
YEARS=(2015 2016)
# YEARS=(2016)
# CORPORA=$DATA_DIR/$YEAR/*clean_normalized.bz2
# SOCIAL_VAR=user
SOCIAL_VAR=subreddit

# to process all files at once
# for YEAR in "${YEARS[@]}";
# do
#     CORPORA=$DATA_DIR/$YEAR/*clean_normalized.bz2
#     for CORPUS in $CORPORA;
#     do
# 	echo $CORPUS
# 	(python get_social_word_counts.py --comment_files $CORPUS --social_vars $SOCIAL_VAR)&
#     done
# done

# to process specific files
# YEAR_DIR=$DATA_DIR/2015
# PREFIX="RC_2015-"
# SUFFIX="_normalized.bz2"
# CORPORA=($YEAR_DIR/"$PREFIX"07"$SUFFIX" $YEAR_DIR/"$PREFIX"08"$SUFFIX" $YEAR_DIR/"$PREFIX"09"$SUFFIX" $YEAR_DIR/"$PREFIX"10"$SUFFIX" $YEAR_DIR/"$PREFIX"11"$SUFFIX")
CORPORA=($DATA_DIR/2015/RC_2015-06_normalized.bz2 $DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2 $DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2 $DATA_DIR/2016/RC_2016-01_normalized.bz2 $DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
for CORPUS in "${CORPORA[@]}";
do
    echo $CORPUS
    (python get_social_word_counts.py --comment_files $CORPUS --social_vars $SOCIAL_VAR)&
done