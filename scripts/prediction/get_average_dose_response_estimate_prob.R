set.seed(123)
library(causaldrf)
library(reshape2)
source('hi_est_logit.R')
# debugging
sink(stdout(), type="message")

## estimating ADRF for given treatment (combo), control and outcome (combo) variables
run_h_estimate <- function(t_vars, c_var, o_vars, timestep_range, stats, outcome_formula_str=''){
    ## Run the Hirano-Imbens estimator for average dose
    ## response, using the specified treatment, control and output.
    
    # if no control vars, we assume that every variable except
    # treatment is control
    c_vars_0 <- c_vars
    if(length(c_vars) > 0){
        c_var_str <- paste(c_vars, collapse=',')
    }
    all_vars <- Filter(function(x) { return(x != 'word')}, colnames(stats))
    quant_probs <- seq(0, 0.99, by=0.01)
    # parameters
    param_df <- data.frame()
    param_colnames <- c('timesteps', 'control_vars', 'treatment_name', 'treatment_coeff', 'gps_coeff', 'intercept_coeff')
    # raw generated values
    vals_df <- data.frame()
    val_colnames <- c('timesteps', 'treatment_name', 'control_vars', quant_probs)
    # default outcome formula = linear
    if(outcome_formula_str == ''){
        outcome_formula_str <- paste('Y', paste(c('T', 'gps'), collapse = '+'), sep = '~')
    }
    
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
            # convert estimator command to text and parse
            hi_est_str <- paste(c("hi_est_logit(Y=Y, treat=T, treat_formula=",treat_formula_str, ",outcome_formula=",outcome_formula_str, ",data=tmp_data, grid_val=grid_val, treat_mod='Normal', link_function='inverse')"), collapse="")
            hi_estimate <- eval(parse(text=hi_est_str))
            hi_coeffs <- hi_estimate[['out_mod']][['coefficients']]
            hi_est_vals <- hi_estimate[['param']]
            param_row <- c(timestep_range, t_var, c_var_str, hi_coeffs[['T']], hi_coeffs[['gps']], hi_coeffs[['(Intercept)']])
            param_row <- t(data.frame(param_row))
            val_row <- c(timestep_range, t_var, c_var_str, hi_est_vals)
            val_row <- t(data.frame(val_row))
            if(length(param_df) == 0){
                param_df <- param_row
                vals_df <- val_row
            }
            else{
                param_df <- as.data.frame(rbind(param_df, param_row))
                vals_df <- as.data.frame(rbind(vals_df, val_row))
            }
        }
    }
    # set colnames, rownames
    colnames(param_df) <- param_colnames
    rownames(param_df) <- 1:dim(param_df)[1]
    colnames(vals_df) <- val_colnames
    rownames(vals_df) <- 1:dim(vals_df)[1]
    results <- list(params=param_df, vals=vals_df)
    return(results)
}
combine_all_stats <- function(stats, stat_names, time_steps) {
    # compute mean for each word between timesteps
    for(s_name in stat_names){
        s <- stats[[s_name]]
        s_flat <- s[, time_steps]
        s_flat[is.na(s_flat)] <- min(s_flat)
        # compute mean over timesteps
        s_flat_df <- data.frame(apply(s_flat, 1, mean), row.names=rownames(s_flat))
        colnames(s_flat_df) <- c(s_name)
        if(length(all_stats_flat) == 0){
            all_stats_flat <- s_flat_df
        }
        else{
            all_stats_flat <- cbind(all_stats_flat, s_flat_df)
        }
    }
    # remove nan values
    all_stats_flat <- all_stats_flat[complete.cases(all_stats_flat), ]
    all_stats_vocab <- as.data.frame(rownames(all_stats_flat))
    colnames(all_stats_vocab) <- c('word')
    # Z-score standardize for ~accurate comparison~
    all_stats_flat <- as.data.frame(apply(all_stats_flat, 2, function(x) {return(scale(x))}))
    all_stats_flat <- t(data.frame(do.call(rbind, all_stats_flat)))
    # add word column
    all_stats_flat <- cbind(all_stats_flat, all_stats_vocab)
    
    return(all_stats_flat)
}

## run the test multiple times
run_bootstrap <- function(t_vars, c_vars, o_vars, timestep_range, data, bootstrap_iters){
    combined_params <- data.frame()
    combined_vals <- data.frame()
    N <- dim(data)[1]
    for(i in 1:bootstrap_iters){
        tmp_data <- data[sample(N, N, replace=TRUE), ]
        results <- run_h_estimate(t_vars, c_vars, o_vars, timestep_range, tmp_data)
        params <- results[['params']]
        vals <- results[['vals']]
        # add bootstrap iter as column
        params[, 'iter'] <- i
        vals[, 'iter'] <- i
        if(dim(combined_params)[1] == 0){
            combined_params <- params
            combined_vals <- vals
        }
        else{
            combined_params <- rbind(combined_params, params)
            combined_vals <- rbind(combined_vals, vals)
        }
#         if(i %% 10 == 0){
#             print(paste('processed ', i, ' bootstrap iters', sep=''))
#         }
    }
    combined_data <- list(params=combined_params, vals=combined_vals)
    return(combined_data)
}
## run test multiple times with balanced classes
run_bootstrap_balanced_class <- function(t_vars, c_vars, o_vars, class_var, timestep_range, data, bootstrap_iters){
    combined_params <- data.frame()
    combined_vals <- data.frame()
    N <- dim(data)[1]
    N_minority <- as.vector(sort(table(data[, class_var]), decreasing = FALSE)[1])[1]
    class_list <- unique(data[, class_var])
    for(i in 1:bootstrap_iters){
        tmp_data <- data[sample(N, N, replace=FALSE), ]
        tmp_data_resampled <- data.frame()
        # sample same number of rows per class
        for(c in class_list){
            tmp_data_c <- data[data[, class_var] == c,]
            N_c <- dim(tmp_data_c)[1]
            tmp_data_c <- tmp_data_c[sample(N_c, N_minority, replace=TRUE), ]
            if(dim(tmp_data_resampled)[1] == 0){
                tmp_data_resampled <- tmp_data_c
            }
            else{
                tmp_data_resampled <- rbind(tmp_data_resampled, tmp_data_c)
            }
        }
        tmp_data <- tmp_data_resampled
        results <- run_h_estimate(t_vars, c_vars, o_vars, timestep_range, tmp_data, outcome_formula_str=outcome_formula_str)
        params <- results[['params']]
        vals <- results[['vals']]
        # add bootstrap iter as column
        params[, 'iter'] <- i
        vals[, 'iter'] <- i
        if(dim(combined_params)[1] == 0){
            combined_params <- params
            combined_vals <- vals
        }
        else{
            combined_params <- rbind(combined_params, params)
            combined_vals <- rbind(combined_vals, vals)
        }
        if(i %% 10 == 0){
            print(paste('processed ', i, ' bootstrap iters', sep=''))
        }
    }
    # cast outcome probabilities as double
    treatment_level_names <- Filter(function(x){return(x != 'timesteps' & x != 'treatment_name' & x != 'control_vars' & x != 'iter')}, colnames(combined_vals))
    combined_vals[, treatment_level_names] <- apply(combined_vals[, treatment_level_names], 2, as.double)
    # combine finally
    combined_data <- list(params=combined_params, vals=combined_vals)
    return(combined_data)
}

## writing results to file
write_results <- function(results, est_str, c_var_str, o_var_str, word_list_name, outcome_formula_type, out_dir){
    # write parameters and estimated outcome values to file
    param_result_file <- file.path(out_dir, paste(c(est_str, c_var_str, o_var_str, word_list_name, outcome_formula_type, 'param_results.tsv'), collapse='_'))
    raw_val_result_file <- file.path(out_dir, paste(c(est_str, c_var_str, o_var_str, word_list_name, outcome_formula_type, 'outcome_results.tsv'), collapse='_'))
    param_df <- results[['params']]
    vals_df <- results[['vals']]
    write.table(param_df, param_result_file, sep='\t')
    write.table(vals_df, raw_val_result_file, sep='\t')
}

args <- commandArgs(trailingOnly=T)
if(length(args) == 0){
    out_dir <- '../../output/'
} else{
    out_dir <- args[1]
}
## load and process data
# load data
data_dir <- '../../data/frequency'
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
# load growth, decline words for later
word_dir <- file.path(data_dir, 'word_lists')
growth_word_file <- file.path(word_dir, '2013_2016_growth_words_clean.csv')
decline_word_file <- file.path(word_dir, '2013_2016_growth_decline_combined.csv')
growth_words <- as.character(read.csv(growth_word_file, sep=',', header=TRUE)[, 'word'])
decline_words <- as.character(read.csv(decline_word_file, header=TRUE)[, 'word'])
change_words <- c(growth_words, decline_words)
# get overlapping vocab
all_stats <- list(f, DL, DU, DS, DT)
all_idx <- sapply(all_stats, row.names)
vocab <- Reduce(intersect, all_idx)
# update change words to ignore stuff not in vocab
change_words <- intersect(change_words, vocab)
# restrict stats to vocab
f <- f[vocab,]
DL <- DL[vocab,]
DU <- DU[vocab,]
DS <- DS[vocab,]
DT <- DT[vocab,]
# combine stats
all_stats <- list(f=f, DL=DL, DS=DS, DU=DU, DT=DT)
all_stats_flat <- data.frame()
stat_names <- names(all_stats)
# time_steps <- 1:12
# timestep_range <- paste(time_steps[1], tail(time_steps,1), sep='_')
# stats_combined <- combine_all_stats(all_stats, stat_names, time_steps)

## run tests

# set treatment, output
est_method <- 'hi_est'
t_vars <- c('DL', 'DU', 'DS', 'DT')
o_vars <- c('growth')
word_list_name <- 'change'
# for balancing classes
class_var <- 'growth'
o_var_str <- paste(o_vars, collapse=',')
# linear outcome
outcome_formula_type <- 'linear'
outcome_formula_str <- paste('Y', paste(c('T', 'gps'), collapse = '+'), sep = '~')
# quadratic outcome
# outcome_formula_type <- 'quad'
# outcome_formula_str <- paste('Y', paste(c('T', 'gps', 'T**2', 'gps**2', 'T*gps'), collapse = '+'), sep = '~')

bootstrap_iters <- 50
timestep_range_list <- list(1:12, 13:24, 25:36, 1:6, 6:12, 12:18)
for(time_steps in timestep_range_list){
    # new sub directory for every timestep range
    timestep_range <- paste(time_steps[1], tail(time_steps,1), sep='_')
    timestep_out_dir <- file.path(out_dir, paste('ADRF_', timestep_range, sep=''))
    if(!file.exists(timestep_out_dir)){
        dir.create(timestep_out_dir, showWarnings = FALSE)
    }
    ## generate stats
    # generate stats using mean over timesteps
    stats_combined <- combine_all_stats(all_stats, stat_names, time_steps)
    stats_combined_change <- stats_combined[mapply(function(x) { return(x %in% change_words)}, stats_combined[, 'word']), ]
    growth_vals <- mapply(function(x) { return(as.integer(x %in% growth_words))}, stats_combined_change[, 'word'])
    growth_vals <- data.frame(growth_vals)
    colnames(growth_vals) <- 'growth'
    stats_combined_change <- cbind(stats_combined_change, growth_vals)
    stats_combined_growth <- stats_combined_change[stats_combined_change[, 'growth']==1, ]
    stats_combined_decline <- stats_combined_change[stats_combined_change[, 'growth']==0, ]
    
    ## run tests
    # (1) use frequency as control; growth/decline words only
    c_vars <- c('f')
    c_var_str <- paste(c_vars, collapse=',')
    f_control_growth_prob_results <- run_bootstrap_balanced_class(t_vars, c_vars, o_vars, class_var, timestep_range, stats_combined_change, bootstrap_iters)
    write_results(f_control_growth_prob_results, est_method, c_var_str, o_var_str, word_list_name, outcome_formula_type, timestep_out_dir)
    # (2) use all covariates as control; growth/decline words only
    c_vars <- c()
    c_var_str <- 'all_control'
    all_control_growth_prob_results <- run_bootstrap_balanced_class(t_vars, c_vars, o_vars, class_var, timestep_range, stats_combined_change, bootstrap_iters)
#     write_results(all_control_growth_prob_results, est_method, c_var_str, o_var_str, word_list_name, out_dir)
    write_results(all_control_growth_prob_results, est_method, c_var_str, o_var_str, word_list_name, outcome_formula_type, timestep_out_dir)
}