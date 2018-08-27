library(stringr)
library(relaimpo)
library(reshape)
library(data.table)
source('prediction_helpers.R')
smooth_stat <- function(x) {
    return(x + min(x[x>0.]))
}
args <- commandArgs(trailingOnly=T)
if(length(args) == 0){
    out_dir <- '../../output/'
} else{
    out_dir <- args[1]
}
data_dir <- '../../data/frequency'

# all covariates
covariate_file_bases <- c('2013_2016_tf_norm_log.tsv', '2013_2016_user_diffusion_log.tsv', '2013_2016_subreddit_diffusion_log.tsv', '2013_2016_thread_diffusion_log.tsv', '2013_2016_3gram_residuals.tsv')
covariate_names <- c('f', 'DU', 'DS', 'DT', 'DL')
# social dissemination only
# covariate_file_bases <- c('2013_2016_tf_norm_log.tsv', '2013_2016_user_diffusion_log.tsv', '2013_2016_subreddit_diffusion_log.tsv', '2013_2016_thread_diffusion_log.tsv')
# covariate_names <- c('f', 'DU', 'DS', 'DT')
# social dissemination subsets
# DU
# covariate_file_bases <- c('2013_2016_tf_norm_log.tsv', '2013_2016_user_diffusion_log.tsv')
# covariate_names <- c('f', 'DU')
# DS
# covariate_file_bases <- c('2013_2016_tf_norm_log.tsv', '2013_2016_subreddit_diffusion_log.tsv')
# covariate_names <- c('f', 'DS')
# DT
# covariate_file_bases <- c('2013_2016_tf_norm_log.tsv', '2013_2016_thread_diffusion_log.tsv')
# covariate_names <- c('f', 'DT')

covariate_files <- file.path(data_dir, covariate_file_bases)

# load default data	
tf <- read.table(paste(data_dir, '2013_2016_tf_norm_log.tsv', sep='/'), sep = '\t', header = TRUE, row.names = 1)
covariates <- lapply(covariate_files, function(f) {
    cov <- read.table(f, sep='\t', quote="", header=TRUE, row.names=1)
    return(cov)
})
names(covariates) <- covariate_names
# optional: collect full vocab for testing
full_vocab <- lapply(covariates, rownames)
full_vocab <- Reduce(intersect, full_vocab)

word_lists <- load_word_lists(data_dir)
growth_words <- word_lists[['growth']]

# combine data
melt_covariates <- function(covariate, covariate_name, vocab) {
        covariate_melted <- melt(as.matrix(covariate[vocab, ]))
        colnames(covariate_melted) <- c('word', 'time', covariate_name)
        return(covariate_melted)
    }

combine_data <- function(covariates, covariate_names, delta_stat, delta_stat_name, vocab, delta_shift) {
    N <- dim(delta_stat)[2]
    K <- N - delta_shift
    all_dates <- colnames(delta_stat)
    shifted_dates <- all_dates[1:K]
    covariates_shifted <- lapply(covariates, function(x) { return(x[, shifted_dates]) })
    covariate_melted <- mapply(function(x,y){return(melt_covariates(x, y, vocab))}, covariates_shifted, covariate_names)
    covariate_melted_vals <- as.data.table(covariate_melted[seq(1, length(covariate_melted), 3) + 2])
    colnames(covariate_melted_vals) <- covariate_names
    word_date_cols <- cbind(as.data.table(covariate_melted[1]), as.data.table(covariate_melted[2]))
    colnames(word_date_cols) <- c('word', 'date')
    covariate_melted_vals <- cbind(word_date_cols, covariate_melted_vals)
    delta <- t(diff(as.matrix(t(delta_stat[vocab, ])), lag=delta_shift))
    colnames(delta) <- shifted_dates
    delta_melted <- melt(as.matrix(delta))
    combined_data <- cbind(covariate_melted_vals, delta_melted[, 'value'])
    setnames(combined_data, 'V2', delta_stat_name)
    combined_data <- as.data.frame(combined_data)
    return(combined_data)
}

delta_stat <- tf
delta_stat_name <- 'f_delta'
# covariates <- list(tf, C3, DU, DT)
# covariate_names <- c('f', 'C3', 'DU', 'DS', 'DT')
# covariate_names <- c('f', 'C3', 'DU_log', 'DT_log')
window_sizes <- c(12,24)
out_file <- paste(out_dir, 'relative_importance_tests.txt', sep='/')
# out_file <- paste(out_dir, 'relative_importance_tests_DU_log_DT_log.txt', sep='/')
# start with clean slate
if(file.exists(out_file)) {
    file.remove(out_file)
}
bootstrap_iters <- 100
combined_bootstrap_results <- data.frame()
significance_results <- data.frame()
relative_importance_methods <- c('lmg')
rand_eval_diff_colnames <- c('diff', 'significance', 'lower', 'upper')
# just growth words
vocab <- growth_words
file_suffix <- ''
# full vocabulary
# vocab <- full_vocab
# file_suffix <- '_full_vocab'
for(window_size in window_sizes) {
    combined_data_windows <- combine_data(covariates, covariate_names, delta_stat, delta_stat_name, vocab, window_size)
    N <- dim(combined_data_windows)[1]
    combined_data_rescaled <- as.data.frame(scale(as.data.frame(combined_data_windows[, c(covariate_names, 'f_delta')])))

    # get relative importance
    relimp_formula <- as.formula(paste('f_delta', paste(covariate_names, collapse=' + '), sep=' ~ '))
    relimp_results <- calc.relimp(formula=relimp_formula, data=combined_data_rescaled, type =relative_importance_methods, rela = F)
    capture.output(paste('testing window size', window_size, sep=' '), file=out_file, append=TRUE)
    capture.output(relimp_results, file=out_file, append=TRUE)
    
    # get bootstrap confidence intervals
    relimp_boot <- boot.relimp(relimp_formula, combined_data_rescaled, b = bootstrap_iters, type=relative_importance_methods, rela = FALSE)
    relimp_eval <- booteval.relimp(relimp_boot)
    relimp_coefficients <- as.data.frame(relimp_eval$mark)
    relimp_coefficients[, 'window'] <- window_size
    capture.output(paste('bootstrap confidence interval'), file=out_file, append=TRUE)
    capture.output(format(relimp_coefficients, digits=8, nsmall=8,scientific=TRUE), file=out_file, append=TRUE)
    # replace with actual useful values
    relimp_coefficients[, '0.95.1'] <- as.vector(relimp_eval$lmg.lower)
    relimp_coefficients[, '0.95.2'] <- as.vector(relimp_eval$lmg.upper)
    relimp_coefficients[, 'percentage'] <- as.vector(relimp_eval$lmg)
    # record bootstrap confidence intervals
    combined_bootstrap_results <- rbind(combined_bootstrap_results, relimp_coefficients)
  
    # significance test: add random variable, compare difference between covariate and random
    rand_name <- 'rand'
    combined_data_rescaled[, rand_name] <- rexp(N)
    covariate_names_rand <- c(covariate_names, rand_name)
    relimp_formula_rand <- as.formula(paste('f_delta', paste(covariate_names_rand, collapse=' + '), sep=' ~ '))
    rand_relimp <- boot.relimp(relimp_formula_rand, combined_data_rescaled, b = bootstrap_iters, type=relative_importance_methods, diff = TRUE, rela = FALSE)
    rand_eval <- booteval.relimp(rand_relimp)
    capture.output(paste('compare with random variable for significance'), file=out_file, append=TRUE)
#     capture.output(attributes(rand_eval), file=out_file, append=TRUE)
#     capture.output(rand_eval, file=out_file, append=TRUE)
    rand_eval_diffs <- as.data.frame(rand_eval$markdiff)
    # reset col names
    colnames(rand_eval_diffs) <- rand_eval_diff_colnames
    rand_eval_diffs[, 'window'] <- window_size
    capture.output(rand_eval_diffs, file=out_file, append=TRUE)
    significance_results <- rbind(significance_results, rand_eval_diffs)

    # get coefficients from regular regression
    lm_results <- lm(formula=relimp_formula, data=combined_data_rescaled)
    capture.output(paste('regression coefficients'), file=out_file, append=TRUE)
    capture.output(summary(lm_results), file=out_file, append=TRUE)
}

# write bootstrap confidence intervals
# bootstrap_out_file <- file.path(out_dir, 'relative_importance_bootstrap_results.tsv')
combined_covariate_names <- paste(covariate_names, collapse='_')
bootstrap_out_file <- file.path(out_dir, paste('relative_importance_coefficients_', combined_covariate_names, file_suffix, '.tsv', sep=''))
# make sure we have enough digits
combined_bootstrap_results[, 'percentage'] <- as.numeric(as.character(combined_bootstrap_results[, 'percentage']))
combined_bootstrap_results[, '0.95.1'] <- as.numeric(as.character(combined_bootstrap_results[, '0.95.1']))
combined_bootstrap_results[, '0.95.2'] <- as.numeric(as.character(combined_bootstrap_results[, '0.95.2']))
# rename
names(combined_bootstrap_results)[names(combined_bootstrap_results) == '0.95.1'] <- '0.95.lower'
names(combined_bootstrap_results)[names(combined_bootstrap_results) == '0.95.2'] <- '0.95.upper'
combined_bootstrap_results <- format(combined_bootstrap_results, digits=8, nsmall=8,scientific=TRUE)
write.table(combined_bootstrap_results, bootstrap_out_file, sep='\t', quote=FALSE)

# write random variable coefficient significance tests
significance_out_file <- file.path(out_dir, paste('relative_importance_coefficient_significance_', combined_covariate_names, file_suffix, '.tsv', sep=''))
write.table(significance_results, significance_out_file, sep='\t', quote=FALSE)