library(causaldrf)
library(reshape2)
# debugging
sink(stdout(), type="message")

args <- commandArgs(trailingOnly=T)
if(length(args) == 0){
    out_dir <- '../../output/'
}
else{
    out_dir <- args[1]
}
## load and process data
# load data
data_dir <- '../../data/frequency'
# out_dir <- '../../output/'
tf_file <- file.path(data_dir, '2013_2016_tf_norm_log.tsv')
DL_file <- file.path(data_dir, '2013_2016_3gram_residuals.tsv')
DU_file <- file.path(data_dir, '2013_2016_user_diffusion.tsv')
DS_file <- file.path(data_dir, '2013_2016_subreddit_diffusion.tsv')
DT_file <- file.path(data_dir, '2013_2016_thread_diffusion.tsv')
f <- read.table(tf_file, sep = '\t', row.names = 1, header = TRUE, check.names = FALSE)
DL <- read.table(DL_file, sep = '\t', row.names = 1, quote = "", header = TRUE, check.names = FALSE)
DU <- read.table(DU_file, sep = '\t', row.names = 1, quote = "", header = TRUE, check.names = FALSE)
DS <- read.table(DS_file, sep = '\t', row.names = 1, quote = "", header = TRUE, check.names = FALSE)
DT <- read.table(DT_file, sep = '\t', row.names = 1, quote = "", header = TRUE, check.names = FALSE)
# load success, failure words for later
word_dir <- file.path(data_dir, 'word_lists')
success_word_file <- file.path(word_dir, '2013_2016_growth_words_clean.csv')
fail_word_file <- file.path(word_dir, '2013_2016_growth_decline_combined.csv')
success_words <- as.character(read.csv(success_word_file, sep=',', header=TRUE)[, 'word'])
fail_words <- as.character(read.csv(fail_word_file, header=TRUE)[, 'word'])
change_words <- c(success_words, fail_words)
print(paste(length(change_words), 'change words', sep=' '))
# get overlapping vocab
all_stats <- list(f, DL, DU, DS, DT)
all_idx <- sapply(all_stats, row.names)
vocab <- Reduce(intersect, all_idx)
# restrict stats to vocab
f <- f[vocab,]
DL <- DL[vocab,]
DU <- DU[vocab,]
DS <- DS[vocab,]
DT <- DT[vocab,]
# combine stats
all_stats <- list(f=f, DL=DL, DS=DS, DU=DU, DT=DT)
s_names <- names(all_stats)
all_stats_flat <- data.frame()
for(s_name in s_names){
    s <- all_stats[[s_name]]
    s_flat <- s
    s_flat[, 'word'] <- rownames(s_flat)
    s_flat <- melt(s_flat, id.var='word', variable.name='time', value.name=s_name)
#     all_stats_flat[, s_name] <- s_flat[, s_name]
    s_flat_vec <- s_flat[, s_name]
    s_flat_df <- data.frame(s_name=s_flat_vec)
    colnames(s_flat_df) <- c(s_name)
    if(length(all_stats_flat) == 0){
        all_stats_flat <- s_flat_df
    }
    else{
        all_stats_flat <- cbind(all_stats_flat, s_flat_df)
    }
}
# replace nan values
all_stats_flat[is.na(all_stats_flat)] <- 0
# add word/time columns to all stats flat
word_time <- all_stats[['f']]
word_time[, 'word'] <- rownames(word_time)
word_time <- melt(word_time, id.var='word', variable.name='time')[, c('word', 'time')]
all_stats_flat <- cbind(all_stats_flat, word_time)
# compute frequency deltas
N <- length(f)
k_vals <- c(N / 6, N / 3, N * (2/3))
stats_combined <- all_stats_flat
for(k in k_vals){
    f_0 <- f[, 1:(N-k)]
    f_k <- f[, (k+1):N]
    f_delta_k <- f_k - f_0
    colnames(f_delta_k) <- colnames(f_0)
    f_delta_k[, 'word'] <- rownames(f_delta_k)
    s_name <- paste('f_delta_', k, sep='')
    f_delta_k_flat <- melt(f_delta_k, id.var='word', variable.name='time', value.name=s_name)
    stats_combined <- merge(stats_combined, f_delta_k_flat, by=c('word', 'time'))
}

## run tests
## TODO: run bootstraps to estimate CI intervals ;_;
run_h_estimate <- function(t_vars, c_vars, o_vars, stats, use_log=FALSE){
    ## Run the Hirano-Imbens estimator for average dose
    ## response, using the specified treatment, control and output.
    
    # if no control vars, we assume that every variable except
    # treatment is control
    c_vars_0 <- c_vars
    if(length(c_vars) > 0){
        c_var_str <- paste(c_vars, collapse=',')
    }
    all_vars <- Filter(function(x) { return(x != 'word' & x != 'time')}, colnames(stats))
    quant_probs <- seq(0, 0.95, by=0.01)
    # parameters
    hi_est_colnames <- c('k', 'treatment_name', 'treatment_coeff', 'gps_coeff', 'intercept_coeff')
    hi_est_df <- data.frame()
    # raw generated values
    hi_est_val_colnames <- c('k', 'treatment_name', 'treatment_coeff', quant_probs)
    hi_est_vals_df <- data.frame()
    
    for(o_var in o_vars){
        for(t_var in t_vars){
            if(length(c_vars_0) == 0){
                c_vars <- Filter(function(x) { return(x != t_var & !(x %in% o_vars))}, all_vars)
                c_var_str <- paste(c_vars, collapse=',')
            }
            tmp_data <- as.data.frame(cbind(stats[, t_var], stats[, c_vars], stats[, o_var]))
            colnames(tmp_data) <- c('T', c_vars, 'Y')
            tmp_data <- tmp_data[complete.cases(tmp_data[, 'Y']), ]
            grid_val <- quantile(tmp_data[,'T'], probs=quant_probs)
            # generate formulae
            treat_formula_str <- paste('T', paste(c_vars, collapse = '+'), sep = '~')
            outcome_formula_str <- paste('Y', paste(c('T', 'gps'), collapse = '+'), sep = '~')
            # convert estimator command to text and parse
            if(use_log){
                hi_est_str <- paste(c("hi_est_logit(Y=Y, treat=T, treat_formula=",treat_formula_str, ",outcome_formula=",outcome_formula_str, ",data=tmp_data, grid_val=grid_val, treat_mod='Normal', link_function='inverse')"), collapse="")
            }
            else{
                hi_est_str <- paste(c("hi_est(Y=Y, treat=T, treat_formula=",treat_formula_str, ",outcome_formula=",outcome_formula_str, ",data=tmp_data, grid_val=grid_val, treat_mod='Normal', link_function='inverse')"), collapse="")
            }
            # run estimator
            hi_estimate <- eval(parse(text=hi_est_str))
            # organize data
            hi_coeffs <- hi_estimate[['out_mod']][['coefficients']]
            hi_est_vals <- hi_estimate[['param']]
            hi_est_row <- c(k, t_var, c_var_str, hi_coeffs[['T']], hi_coeffs[['gps']], hi_coeffs[['(Intercept)']])
            hi_est_row <- t(data.frame(hi_est_row))
            hi_est_val_row <- c(k, t_var, c_var_str, hi_est_vals)
            hi_est_val_row <- t(data.frame(hi_est_val_row))
            # append to dataframes
            if(length(hi_est_df) == 0){
                hi_est_df <- hi_est_row
                hi_est_vals_df <- hi_est_vals
            }
            else{
                hi_est_df <- as.data.frame(rbind(hi_est_df, hi_est_row))
                hi_est_vals_df <- as.data.frame(rbind(hi_est_vals_df, hi_est_vals))
            }
        }
    }
    # set colnames, rownames
    colnames(hi_est_df) <- hi_est_colnames
    rownames(hi_est_df) <- 1:dim(hi_est_df)[1]
    colnames(hi_est_vals_df) <- hi_est_val_colnames
    rownames(hi_est_vals_df) <- 1:dim(hi_est_vals_df)[1]
    hi_est_data <- list(params=hi_est_df, vals=hi_est_vals_df)
    return(hi_est_data)
}
write_results <- function(results, est_str, c_vars, o_var_str, word_list_name, out_dir){
    # write parameters and estimated outcome values to file
    if(length(c_vars) == 0){
        c_var_str <- 'all_control'
    }
    else{
        c_var_str <- paste(paste(c_vars, collapse=','), '_control', sep='')
    }
    param_result_file <- file.path(out_dir, paste(c(est_str, c_var_str, o_var_str, word_list_name, 'param_results.tsv'), collapse='_'))
    raw_val_result_file <- file.path(out_dir, paste(c(est_str, c_var_str, o_var_str, word_list_name, 'outcome_results.tsv'), collapse='_'))
    param_df <- results[['params']]
    vals_df <- results[['vals']]
    write.table(param_df, param_result_file, sep='\t')
    write.table(vals_df, raw_val_result_file, sep='\t')
}
# set treatment, output
t_vars <- c('DL', 'DU', 'DS', 'DT')
o_vars <- paste('f_delta_', k_vals, sep='')
# TODO: iterate over conditions automatically
c_var_lists <- list(c('f'), c())
o_var_list <- list(o_vars, c('success'))
stats_combined_change <- stats_combined[mapply(function(x) { return(x %in% change_words)}, stats_combined[, 'word']), ]
# add success condition
stats_combined_change[, 'success'] <- mapply(function(x) { return(as.integer(x %in% success_words))}, stats_combined_change[, 'word'])
stats_combined_success <- stats_combined_change[stats_combined_change[, 'success']==1, ]
stats_combined_fail <- stats_combined_change[stats_combined_change[, 'success']==0, ]
stat_list <- list(stats_combined, stats_combined_change, stats_combined_success, stats_combined_fail)
est_method <- 'hi_est'
o_var_name <- 'f_delta'
word_list_name <- 'all_words'
# (1) use frequency as control
print('test 1')
c_vars <- c('f')
f_control_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined)
# f_control_out_file <- file.path(out_dir, 'hi_est_f_control_results.tsv')
write_results(f_control_results, est_method, c_vars, o_var_name, word_list_name, out_dir)
# write.table(f_control_results, f_control_out_file, sep='\t')
# (2) use all covariates as control
print('test 2')
# pass in an empty list to trigger all-covariate condition
c_vars <- c()
all_control_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined)
write_results(all_control_results, est_method, c_vars, o_var_str, word_list_name, out_dir)
# all_control_out_file <- file.path(out_dir, 'hi_est_all_control_results.tsv')
# write.table(all_control_results, all_control_out_file, sep='\t')
# (3) use frequency as control; success/failure words only
print('test 3')
c_vars <- c('f')
word_list_name <- 'change_words'
f_control_change_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined_change)
write_results(f_control_change_results, est_method, c_vars, o_var_str, word_list_name, out_dir)
# f_control_change_out_file <- file.path(out_dir, 'hi_est_f_control_change_results.tsv')
# write.table(f_control_change_results, f_control_change_out_file, sep='\t')
# (4) use all covariates as control; success/failure words only
print('test 4')
c_vars <- c()
all_control_change_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined_change)
write_results(all_control_change_results, est_method, c_vars, o_var_str, word_list_name, out_dir)
# all_control_change_out_file <- file.path(out_dir, 'hi_est_all_control_change_results.tsv')
# write.table(all_control_change_results, all_control_change_out_file, sep='\t')
# (5) use frequency as control; success words only
print('test 5')
c_vars <- c('f')
word_list_name <- 'success_words'
f_control_success_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined_success)
write_results(f_control_success_results, est_method, c_vars, o_var_str, word_list_name, out_dir)
# f_control_success_out_file <- file.path(out_dir, 'hi_est_f_control_success_results.tsv')
# write.table(f_control_success_results, f_control_success_out_file, sep='\t')
# (6) use all covariates as control; success words only
print('test 6')
c_vars <- c()
all_control_success_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined_success)
write_results(all_control_success_results, est_method, c_vars, o_var_str, word_list_name, out_dir)
# all_control_success_out_file <- file.path(out_dir, 'hi_est_all_control_success_results.tsv')
# write.table(all_control_success_results, all_control_success_out_file, sep='\t')
# (7) use frequency as control; fail words only
print('test 7')
c_vars <- c('f')
word_list_name <- 'fail_words'
f_control_fail_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined_fail)
write_results(f_control_fail_results, est_method, c_vars, o_var_str, word_list_name, out_dir)
# f_control_fail_out_file <- file.path(out_dir, 'hi_est_f_control_fail_results.tsv')
# write.table(f_control_fail_results, f_control_fail_out_file, sep='\t')
# (8) use all covariates as control; fail words only
print('test 8')
c_vars <- c()
all_control_fail_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined_fail)
write_results(all_control_fail_results, est_method, c_vars, o_var_str, word_list_name, out_dir)
# all_control_fail_out_file <- file.path(out_dir, 'hi_est_all_control_fail_results.tsv')
# write.table(all_control_fail_results, all_control_fail_out_file, sep='\t')

## same thing as (3-4) but with binary outcome
# (3a) use frequency as control; success/failure words only
print('bout to try binary outcome')
o_vars <- c('success')
o_var_str <- 'success'
word_list_name <- 'change'
c_vars <- c('f')
f_control_success_prob_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined_change)
write_results(f_control_success_prob_results, est_method, c_vars, o_var_str, word_list_name, out_dir)
# f_control_success_prob_file <- file.path(out_dir, 'hi_est_f_control_prob_success_results.tsv')
# write.table(f_control_success_prob_results, f_control_success_prob_file, sep='\t')
# (4a) use all covariates as control; success/failure words only
c_vars <- c()
all_control_success_prob_results <- run_h_estimate(t_vars, c_vars, o_vars, stats_combined_change)
write_results(all_control_success_prob_results, est_method, c_vars, o_var_str, word_list_name, out_dir)
# all_control_success_prob_file <- file.path(out_dir, 'hi_est_all_control_prob_success_results.tsv')
# write.table(all_control_success_prob_results, all_control_success_prob_file, sep='\t')