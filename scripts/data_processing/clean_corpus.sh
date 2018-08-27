# extract and whitespace raw text from corpus
# and then normalize => top-500 subreddits, non-spam users, top-100K vocab
# plus extract metadata

# YEARS=('2015' '2016')
# YEARS=('2014' '2015' '2016')
YEARS=('2014')
# external hard drive
#DATA_DIR='/Volumes/Seagate_Expansion_Drive/reddit_comments'
# adaptation
# DATA_DIR=/hg191/corpora/reddit_comment_data/monthly_submission
# conair
DATA_DIR=/hg190/corpora/reddit_comment_data/monthly_submission
# one file at a time
# INPUT="$DATA_DIR"$"2016/RC_2016-05.bz2"
# echo $INPUT
# CLEAN_INPUT=$DATA_DIR$"2016/RC_2016-05_clean.txt"
# bzcat "$INPUT" | jq .body | sed -e 's/^"//' -e 's/"$//' -e 's/\\n/ /g' -e 's/\"/ /g' -e 's,https*://\S*,,g' -e 's/\./ \./g' -e 's/,/ ,/g' -e 's/?/ ?/g' -e 's/!/ !/g' -e 's/;/ ;/g' -e 's/:/ :/g' -e 's/(/( /g' -e 's/)/ )/g' | tr '[:upper:]' '[:lower:]' > "$CLEAN_INPUT"

# all files at once
for YEAR in "${YEARS[@]}";
do
    # COMMENT_FILES=$DATA_DIR/$YEAR/RC_$YEAR$"-[0-9][0-9]".bz2
    # COMMENT_FILES=$DATA_DIR/$YEAR/RC_$YEAR-[01][0-9]_filtered.bz2
    
    # echo $COMMENT_FILES
    # for custom file list
    # COMMENT_FILES=($DATA_DIR/$YEAR/RC_$YEAR-02.bz2 $DATA_DIR/$YEAR/RC_$YEAR-03.bz2 $DATA_DIR/$YEAR/RC_$YEAR-04.bz2 $DATA_DIR/$YEAR/RC_$YEAR-05.bz2)
    COMMENT_FILES=($DATA_DIR/2013/RC_2013-01_filtered.bz2 $DATA_DIR/2013/RC_2013-02_filtered.bz2 $DATA_DIR/2013/RC_2013-03_filtered.bz2 $DATA_DIR/2013/RC_2013-04_filtered.bz2 $DATA_DIR/2013/RC_2013-05_filtered.bz2 $DATA_DIR/2013/RC_2013-06_filtered.bz2 $DATA_DIR/2013/RC_2013-07_filtered.bz2 $DATA_DIR/2013/RC_2013-08_filtered.bz2 $DATA_DIR/2013/RC_2013-09_filtered.bz2 $DATA_DIR/2013/RC_2013-10_filtered.bz2 $DATA_DIR/2013/RC_2013-11_filtered.bz2 $DATA_DIR/2013/RC_2013-12_filtered.bz2 $DATA_DIR/2014/RC_2014-01_filtered.bz2 $DATA_DIR/2014/RC_2014-02_filtered.bz2 $DATA_DIR/2014/RC_2014-03_filtered.bz2 $DATA_DIR/2014/RC_2014-04_filtered.bz2 $DATA_DIR/2014/RC_2014-05_filtered.bz2)
#    COMMENT_FILES=($DATA_DIR/2014/RC_2014-06_filtered.bz2 $DATA_DIR/2014/RC_2014-07_filtered.bz2 $DATA_DIR/2014/RC_2014-08_filtered.bz2 $DATA_DIR/2014/RC_2014-09_filtered.bz2 $DATA_DIR/2014/RC_2014-10_filtered.bz2 $DATA_DIR/2014/RC_2014-11_filtered.bz2 $DATA_DIR/2014/RC_2014-12_filtered.bz2 $DATA_DIR/2015/RC_2015-01_filtered.bz2 $DATA_DIR/2015/RC_2015-02_filtered.bz2 $DATA_DIR/2015/RC_2015-03_filtered.bz2 $DATA_DIR/2015/RC_2015-04_filtered.bz2 $DATA_DIR/2015/RC_2015-05_filtered.bz2)
    for INPUT in "${COMMENT_FILES[@]}";
    # for INPUT in $COMMENT_FILES;
    do
	# echo $INPUT
	# replace zip suffix with text suffix
	# CLEAN_BASE=${INPUT//\.bz2/}$"_clean"
	CLEAN_BASE=${INPUT//_filtered\.bz2/}$"_clean"
	# echo $CLEAN_BASE
	CLEAN_INPUT=$CLEAN_BASE.txt
	OUT_BASE=$(basename $CLEAN_BASE)
	OUTPUT=../../output/"$OUT_BASE"_output.txt
	# (bzcat "$INPUT" | jq .body | sed -e 's/^"//' -e 's/"$//' -e 's/\\n/ /g' -e 's/\"/ /g' -e 's,https*://\S*,,g' -e 's/\./ \./g' -e 's/,/ ,/g' -e 's/?/ ?/g' -e 's/!/ !/g' -e 's/;/ ;/g' -e 's/:/ :/g' -e 's/(/( /g' -e 's/)/ )/g' -e 's,\([a-zA-Z]\)\1\1\1*,\1\1\1,g' -e 's,r/.*,r/SUB,g' -e 's,u/.*,u/USER,g' > $CLEAN_INPUT && bzip2 -c $CLEAN_INPUT > $CLEAN_BASE.bz2 && rm $CLEAN_INPUT && python normalize_corpus.py $CLEAN_BASE.bz2 > $OUTPUT)&
	(bzcat "$INPUT" | jq .body | sed -e 's/^"//' -e 's/"$//' -e 's/\\n/ /g' -e 's/\"/ /g' -e 's,https*://\S*,,g' -e 's/\./ \./g' -e 's/,/ ,/g' -e 's/?/ ?/g' -e 's/!/ !/g' -e 's/;/ ;/g' -e 's/:/ :/g' -e 's/(/( /g' -e 's/)/ )/g' -e 's,\([a-zA-Z]\)\1\1\1*,\1\1\1,g' -e 's,r/.*,r/SUB,g' -e 's,u/.*,u/USER,g' > $CLEAN_INPUT && bzip2 -c $CLEAN_INPUT > $CLEAN_BASE.bz2 && rm $CLEAN_INPUT > $OUTPUT)&
	# for removing duplicate characters but gets tripped by Unicode on Mac...WHY
	# sed 's/\([A-Za-z]\)\1\1\1*/\1\1\1/g' 
	# (bzip2 $CLEAN_INPUT > $CLEAN_BASE.bz2)&
	# rm $CLEAN_INPUT
	# gzip takes too much CPU when we run 12 in parallel ;_;
	# CLEAN_FNAME=$(basename $CLEAN_BASE)
	# NOTE: set "sed -re 's,([a-zA-Z])\1\1\1,\1\1\1,g'" for Unix, "sed 's,\([a-zA-Z]\)\1\1\1,\1\1\1,g'" for Mac
# | tr '[:upper:]' '[:lower:]'
	# if [ -f $CLEAN_INPUT ]; then
	#     (python normalize_corpus.py $CLEAN_INPUT)&
	    # python normalize_corpus.py $CLEAN_INPUT
	# fi
	# NORM_INPUT=$CLEAN_BASE$"_normalized.bz2"
	# (bunzip2 $NORM_INPUT)&
	
    done
done
