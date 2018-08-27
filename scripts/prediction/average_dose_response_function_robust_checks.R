## robustness checks for prediction using ADRF
## (1) time windows (k=12 over all possible start time steps)
## (2) POS tags (using time window 1:12, can we predict success/fail with just POS tags?)
source('prediction_helpers.R')

window_test <- function(all_stats, stat_names, data, window_length){
    o_var <- 'success'
k <- 10
# balance data 
balanced_data <- balance_data(stats_combined_all_windows, o_var)
acc_df <- data.frame()
for(t_var in t_vars){
    c_vars <- Filter(function(x){return(x != t_var)}, stat_names)
    acc <- k_fold_test(balanced_data, t_var, o_var, c_vars, k=k) * 100
    acc_ <- data.frame(t_var, acc)
    colnames(acc_) <- c('Treatment', 'Accuracy')
    if(dim(acc_df)[1] == 0){
        acc_df <- acc_
    }
    else{
        acc_df <- rbind(acc_df, acc_)
    }
}
acc_agg <- aggregate(Accuracy ~ Treatment, acc_df, c)
acc_names <- acc_agg[, 'Treatment']
acc_agg <- acc_agg[, 'Accuracy']
acc_list <- setNames(split(acc_agg, seq(nrow(acc_agg))), acc_names)
print(acc_list)
}

## load data
data_dir <- '../../data/frequency'
window_length <- 12
k_start <- 1
time_steps <- k_start:(window_length+k_start-1)
out_dir <- '../../output'
t_vars <- c('DL', 'DS', 'DU', 'DT')
input_vars <- c('f')
all_stats <- load_all_stats()
stat_names <- names(all_stats)
word_lists <- load_change_vocab()
success_words <- word_lists[['success_words']]
fail_words <- word_lists[['fail_words']]
change_words <- word_lists[['change_words']]

## (1) window test
window_test(all_stats, stat_names, stats_combined_change, window_length)

## (2) POS test
# combine
stats_combined <- combine_all_stats(all_stats, stat_names, time_steps)
# add success condition
stats_combined_change <- stats_combined[mapply(function(x) { return(x %in% change_words)}, stats_combined[, 'word']), ]
success_vals <- mapply(function(x) { return(as.integer(x %in% success_words))}, stats_combined_change[, 'word'])
success_vals <- data.frame(success_vals)
colnames(success_vals) <- 'success'
stats_combined_change <- cbind(stats_combined_change, success_vals)
# POS tags
pos_tag_file <- file.path(data_dir, '2013_2016_tag_estimates.tsv')
pos_tags <- read.table(pos_tag_file, sep='\t', row.names = 1)
pos_tags <- pos_tags[, 'POS']