# get social diffusion in parallel for all dates
DATES=("2015-06" "2015-07" "2015-08" "2015-09" "2015-10" "2015-11" "2015-12" "2016-01" "2016-02" "2016-03" "2016-04" "2016-05")
#DATES=("2015-06")
# SOCIAL_VAR="user"
SOCIAL_VAR="subreddit"
# SOCIAL_VAR="thread"
for DATE in "${DATES[@]}";
do
    echo $DATE
    (python get_social_dissemination.py --all_dates $DATE)&
done
