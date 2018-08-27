# extract metadata from raw comments
# to use in later analysis
source vars.cfg
for YEAR in "${YEARS[@]}"; do
    COMMENT_FILES=$DATA_DIR/$YEAR/RC_$YEAR$"-[0-9][0-9]".bz2
    for COMMENT_FILE in "${COMMENT_FILES[@]}"; do
	META_FILE=${COMMENT_FILE//\.bz2/_meta.bz2}
	echo $META_FILE
	(bzcat $COMMENT_FILE | jq '.id + "," + .parent_id + "," + .created_utc + "," + .subreddit + "," + .author  + "," + .score' >> $META_FILE)&
	# bzip -c $META_FILE
    done
done
