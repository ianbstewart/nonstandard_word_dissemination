source vars.cfg
YEARS=( "2015" "2016" )
SAMPLE_SIZE=2000000
for YEAR in $YEARS;
do
    CORPORA=$DATA_DIR/$YEAR/*_clean_normalized.bz2
    for CORPUS in $CORPORA;
    do
	# echo $CORPUS
	(python sample_text.py --original_file $CORPUS --sample_size=$SAMPLE_SIZE)&
    done
done

CORPORA=($DATA_DIR/2014/RC_2014-06_normalized.bz2 $DATA_DIR/2014/RC_2014-07_normalized.bz2 $DATA_DIR/2014/RC_2014-08_normalized.bz2 $DATA_DIR/2014/RC_2014-09_normalized.bz2 $DATA_DIR/2014/RC_2014-10_normalized.bz2 $DATA_DIR/2014/RC_2014-11_normalized.bz2 $DATA_DIR/2014/RC_2014-12_normalized.bz2 $DATA_DIR/2015/RC_2015-01_normalized.bz2 $DATA_DIR/2015/RC_2015-02_normalized.bz2 $DATA_DIR/2015/RC_2015-03_normalized.bz2 $DATA_DIR/2015/RC_2015-04_normalized.bz2 $DATA_DIR/2015/RC_2015-05_normalized.bz2 $DATA_DIR/2015/RC_2015-06_normalized.bz2 $DATA_DIR/2015/RC_2015-07_normalized.bz2 $DATA_DIR/2015/RC_2015-08_normalized.bz2 $DATA_DIR/2015/RC_2015-09_normalized.bz2 $DATA_DIR/2015/RC_2015-10_normalized.bz2 $DATA_DIR/2015/RC_2015-11_normalized.bz2 $DATA_DIR/2015/RC_2015-12_normalized.bz2 $DATA_DIR/2016/RC_2016-01_normalized.bz2 $DATA_DIR/2016/RC_2016-02_normalized.bz2 $DATA_DIR/2016/RC_2016-03_normalized.bz2 $DATA_DIR/2016/RC_2016-04_normalized.bz2 $DATA_DIR/2016/RC_2016-05_normalized.bz2)
for CORPUS in $CORPORA;
do
	# echo $CORPUS
    (python sample_text.py --original_file $CORPUS --sample_size=$SAMPLE_SIZE)&
done

# combine into MEGA-SAMPLE
# COMBINED=$DATA_DIR/2015/2015_2016_clean_normalized_sample.txt
# SAMPLES=$DATA_DIR/*/*_clean_normalized_sample.txt
# cat $SAMPLES > $COMBINED
