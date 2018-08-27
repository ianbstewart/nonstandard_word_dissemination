DATA_DIR=../../data/frequency
STATS=($DATA_DIR/2013_2016_user_diffusion.tsv $DATA_DIR/2013_2016_subreddit_diffusion.tsv $DATA_DIR/2013_2016_thread_diffusion.tsv)
for STAT in "${STATS[@]}";
do
#     echo $STAT
    (python get_log_stat.py $STAT)&
done
