# helper methods for binary prediction for growth/decline
# includes matching, combining covariates
library(optmatch)
library(data.table)
library(caret)
library(ROCR)
source('hi_est_logit.R')

## load data
load_word_lists <- function(data_dir='../../data/frequency') {
  growth_words <- as.character(read.csv(file.path(data_dir, 'word_lists/2013_2016_growth_words_clean_final.csv'), header = TRUE)[, 'word'])
  decline_piecewise <- as.character(read.csv(file.path(data_dir, 'word_lists/2013_2016_piecewise_growth_decline_words.csv'), header = TRUE)[, 'word'])
  decline_logistic <- as.character(read.csv(file.path(data_dir, 'word_lists/2013_2016_logistic_growth_decline_words.csv'), header = TRUE)[, 'word'])
  decline_logistic <- setdiff(decline_logistic, decline_piecewise)
  decline_words <- c(decline_piecewise, decline_logistic)
    
  # remove intersecting words
  intersect_words <- intersect(growth_words, decline_words)
  growth_words <- Filter(function(x){return(! x %in% intersect_words)}, growth_words)
  decline_logistic <- Filter(function(x){return(! x %in% intersect_words)}, decline_logistic)
  decline_piecewise <- Filter(function(x){return(! x %in% intersect_words)}, decline_piecewise)
  split_points_piecewise <- read.table(file.path(data_dir, '2013_2016_tf_norm_2_piecewise.tsv'), sep = '\t',
                                       header = TRUE, row.names = 1)
  split_points_logistic <- read.table(file.path(data_dir, '2013_2016_tf_norm_logistic_params.tsv'), sep = '\t',
                                      header = TRUE, row.names = 1)
  split_points <- ceiling(c(split_points_piecewise[decline_piecewise, "x0"], split_points_logistic[decline_logistic, "loc"]))
  decline_words <- c(decline_piecewise, decline_logistic)
  split_points <- data.frame(split_points, row.names = decline_words)
  colnames(split_points) <- c('s')
  # remove words with impossible split points
  MIN_SPLIT=1
  MAX_SPLIT=34
  split_points <- split_points[split_points[, 's'] >= MIN_SPLIT & split_points[, 's'] <= MAX_SPLIT, , drop=FALSE]
  decline_words <- rownames(split_points)
#   split_points <- c(split_points, rep(0, length(growth_words)))
#   split_points <- as.data.frame(split_points, row.names = combined_words)
#   colnames(split_points) <- c("s")
  word_lists <- list(growth=growth_words, decline=decline_words, split=split_points)
  return(word_lists)
}

load_covariates <- function(data_dir) {
  f <- read.table(file.path(data_dir, '2013_2016_tf_norm_log.tsv'), sep='\t', quote="", row.names = 1, header = TRUE)
  C3 <- read.table(file.path(data_dir, '2013_2016_3gram_residuals.tsv'), sep = '\t', quote = "", row.names = 1, header = TRUE)
  D_U <- read.table(file.path(data_dir, '2013_2016_user_diffusion_log.tsv'), sep = '\t', quote = "", row.names = 1, header = TRUE)
  D_S <- read.table(file.path(data_dir, '2013_2016_subreddit_diffusion_log.tsv'), sep = '\t', quote = "", row.names = 1, header = TRUE)
  D_T <- read.table(file.path(data_dir, '2013_2016_thread_diffusion_log.tsv'), sep = '\t', quote = "", row.names = 1, header = TRUE)
  D_U[is.na(D_U)] <- 0.
  D_S[is.na(D_U)] <- 0.
  D_T[is.na(D_U)] <- 0.
  covariates <- list(f=f, C3=C3, D_U=D_U, D_S=D_S, D_T=D_T)
  return(covariates)
}

get_match_stat <- function(f, combined_words, split_points){
  # reorganize data into matrix for matching
  match_stat <- f[combined_words, ]
  T <- dim(match_stat)[2]
  time_range <- 0:(T-1)
  time_steps <- paste('t_', time_range, sep='')
  colnames(match_stat) <- time_steps
  M <- cbind(match_stat, split_points)
  # add growth/no (counts as "treatment" in matching)
  growth <- as.integer(M$s == 0.0)
  M <- cbind(M, growth)
  return(M)
}

## matching
get_dist <- function(w1, w2, M, timesteps) {
    return((sum((M[w1, timesteps] - M[w2, timesteps])**2.))**.5)
}
match_optimal <- function(M, s, k, method = 'mahalanobis') {
    decline_count <- dim(M[M[, 's'] == s, ])[1]
    M[, 'treatment'] <- 1 - M[, 'growth']
    timesteps <- paste('t', (s-k-1):(s-1), sep='_')
    match_formula <- as.formula(paste('treatment', paste(timesteps, collapse = '+'), sep = ' ~ '))
    # compute distances and match
    distances <- match_on(match_formula, data=M, method=method)
    # if only one treatment variable, need to duplicate for some reason
    if(decline_count == 1){
        M_tmp <- rbind(M, M[M[, 's'] == s, ])
        control_name <- rownames(M[M[, 's'] == s, ])
        distances <- match_on(match_formula, data=M_tmp, method=method)
        distances <- t(as.data.frame(distances[1, ]))
        rownames(distances) <- control_name
    }
    match_matrix <- pairmatch(distances, data=M)
    # reorganize matches
    match_list <- match_matrix[complete.cases(match_matrix)]
    match_vals <- as.vector(match_list)
    match_names <- names(match_list)
    match_list <- setNames(match_vals, match_names)
    split_list <- split(match_list, match_vals)
    match_pairs <- t(rbind.data.frame(lapply(split_list, function(x) { return(names(x))})))
    colnames(match_pairs) <- c('word', 'match')
    rownames(match_pairs) <- 1:length(split_list)
    # normalize duplicate words
    match_pairs[, 'word'] <- unlist(lapply(match_pairs[, 'word'], function(x) {return(strsplit(x, split='\\.')[[1]][1])}))
    diff <- apply(match_pairs, 1, function(r){ return(get_dist(r[1], r[2], M, timesteps))})
    match_pairs <- cbind.data.frame(match_pairs, s, k, diff)
    return(match_pairs)
}
get_match_sample_optimal <- function(M, decline_words, growth_words, k, sample_size, method = 'mahalanobis', verbose = TRUE) {
    # Sample the decline words with replacement, match each decline word f to a growth word 
    # using the optimal matching method with data between split point s and k months prior.
      # M = "match matrix"
      # decline_words = "vector of decline words"
      # growth_words = "vector of growth words"
      # k = "months of data prior to split point to use"
      # sample_size = "size of decline word sample"
      # method = "mahalanobis" # distance metric for optimal matching
    decline_sample <- sample(decline_words, sample_size, replace = TRUE)
    combined_words <- c(decline_sample, growth_words)
    if(verbose){
        print(paste(length(combined_words), ' combined words in match sample', sep=''))
    }
    M_sample <- M[combined_words, ]
    S_unique <- sort(unique(M_sample[M_sample[, 's'] > k, ][, 's']))
    match_matrix <- data.frame()
    for(s in S_unique) {
        M_s <- M_sample[M_sample[, 's'] == s | M_sample[, 's'] == 0.0, ]
        match_pairs <- match_optimal(M_s, s, k, method = method)
        if(verbose){
            print(paste('s=', s, sep=''))
            print(paste('got ', dim(M_sample[M_sample[, 's'] == s, ])[1], ' match pairs', sep=''))
            print(head(match_pairs))
        }
        match_matrix <- rbind(match_matrix, match_pairs)
    }
    return(match_matrix)
}


combine_covariate_data <- function(k, matches, covariates, covariate_names, verbose=TRUE){
    covariate_count <- length(covariates)
    all_covariates <- apply(match_matrix, 1, function(match_row) {
        match_words <- as.vector(match_row[c('word', 'match')])
  s <- as.integer(match_row['s'])
  timesteps <- paste('t_', (s-k-1):(s-1), sep='')
  covariate_vals_matches <- mapply(function(cov, cov_name) {
            cov <- cov[match_words, timesteps]
      if(k < 1){
                cov <- as.data.frame(cov)
                row.names(cov) <- match_words
            }
            timestep_names <- 1:(k+1)
            colnames(cov) <- timestep_names
            return(cov)
        }, covariates, covariate_names, SIMPLIFY = FALSE)
  print('got separate covariate vals matches')
        covariate_vals_matches <- do.call("cbind", covariate_vals_matches)
  print('got combined covariate vals matches')
        if(k < 1){
            colnames(covariate_vals_matches) <- covariate_names
        }
  # add Y column
  Y <- c(0,1)
  covariate_vals_matches <- cbind(covariate_vals_matches, Y)
        return(covariate_vals_matches)
    })
    all_covariates <- do.call("rbind", all_covariates)
    print('got all covariates')
    print(head(all_covariates))
    # CAN'T REPLACE ROW NAMES DIRECTLY because of duplicates ;_;
    all_covariates_rows <- as.vector(sapply(row.names(all_covariates), function(x) {return(str_split(x, '\\.')[[1]][2])}))
    all_covariates_rows <- make.names(all_covariates_rows, unique = TRUE)
    row.names(all_covariates) <- all_covariates_rows
    return(all_covariates)
}

prediction_test <- function(data, folds=0, formula=NULL){
    # Predict growth/decline with X/Y data.
      # data = "data frame with differenced covariates and growth/decline output Y"
      # folds = "number of folds to use in cross-validation"
      # formula = "optional prebaked regression formula"
    X_cols <- colnames(data)
    X_cols <- X_cols[X_cols != 'Y']
    
    # full model fit
    if(is.null(formula)){
      formula <- as.formula(paste("Y ~ ", paste(X_cols, collapse='+'), sep=''))  
    }
    model <- glm(formula, data = data, family=binomial(link='logit'))
    model_summary <- summary(model)

    # accuracy on train/test
    if(folds == 0){
        N <- dim(data)[1]
        folds <- N - 2
    }
    tc <- trainControl("cv", folds, savePred=TRUE)
    # change outcome to factor
    data[, 'Y'] <- as.factor(data[, 'Y'])
    print('bout to train')
    fit <- train(Y~., data=data, method='glm', trControl=tc, family=binomial(link='logit'))
    fit_results <- fit$results
    prediction_results <- list(logit=model_summary, accuracy=fit_results)
    return(prediction_results)
}

bootstrap_prediction_test <- function(growth_words, decline_words, 
                                      M, k, feature_sets, feature_set_names, 
                                      sample_size, match_method='mahalanobis', 
                                      bootstrap_iters=10, folds=10) {
    iter_list <- 1:bootstrap_iters
    feature_set_accuracy <- data.frame()
    feature_set_coefficients <- data.frame()
    feature_set_count <- length(feature_sets)
    for(f in 1:feature_set_count){
      covariates <- feature_sets[[f]]
      covariate_names <- names(covariates)
      covariate_count <- length(covariates)
      print(paste('testing covariates', paste(covariate_names, collapse=',')))
      for(i in 1:length(covariates)) {
          covariate <- covariates[[i]]
          covariate <- covariate[, sort(colnames(covariate))]
          timesteps <- paste('t_', 1:dim(covariate)[2], sep='')
          colnames(covariate) <- time_steps
          covariates[[i]] <- covariate
      }
      feature_set_name <- feature_set_names[f]
      prediction_coefficients <- data.frame()
      accuracy_results <- data.frame()
      for(b in iter_list){
          print(paste('starting bootstrap iteration ', b, sep=''))
          match_matrix <- get_match_sample_optimal(M, decline_words, growth_words, k, sample_size, method = match_method, verbose = FALSE)
          # print(paste('got match matrix'))
          # print(head(match_matrix))
          combined_data <- combine_covariate_data(k, match_matrix, covariates, covariate_names, verbose = verbose)
          print('got combined data')
          print(head(combined_data))
          prediction_results <- prediction_test(combined_data, folds=folds)
          coefficients <- as.data.frame(prediction_results$logit$coefficients)
    coefficients[, 'coefficient'] <- rownames(coefficients)
          accuracy <- prediction_results$accuracy
          prediction_coefficients <- rbind(prediction_coefficients, coefficients)
          accuracy_results <- rbind(accuracy_results, accuracy)
      }
      # for each feature set, print mean accuracy
      accuracy_results_relevant <- accuracy_results[, 2:5]
      accuracy_result_mean <- apply(accuracy_results_relevant, 2, mean)
      accuracy_result_mean['AccuracySD'] <- accuracy_result_mean['AccuracySD'] / bootstrap_iters
      print(feature_set_name)
      print(accuracy_result_mean)

  #    feature_set_accuracy[feature_set_name] <- accuracy_result_mean
  #    feature_set_accuracy[feature_set_name] <- accuracy_results_relevant

      # add feature set name to data frame, add bootstrap iters, and save results
      rep_count <- covariate_count*(k+1) + 1
      iter_list_duplicates <- unlist(lapply(iter_list, function(x){ return(rep(x, rep_count)) }))
      print(iter_list_duplicates)
      accuracy_results_relevant <- cbind(accuracy_results_relevant, list(feature_set_name=feature_set_name, folds=folds))
      prediction_coefficients <- cbind(prediction_coefficients, list(feature_set_name=feature_set_name))
      accuracy_results_relevant <- cbind(accuracy_results_relevant, list(iter=iter_list))
      prediction_coefficients <- cbind(prediction_coefficients, list(iter=iter_list_duplicates))
      feature_set_accuracy <- rbind(feature_set_accuracy, accuracy_results_relevant)
      feature_set_coefficients <- rbind(feature_set_coefficients, prediction_coefficients)
  }
    feature_set_values <- list(accuracy=feature_set_accuracy, coefficients=feature_set_coefficients)
    return(feature_set_values)
}
## ADRF prediction
combine_all_stats <- function(stats, stat_names, time_steps, rescale=TRUE) {
    # compute mean for each word between timesteps
    all_stats_flat <- data.frame()
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
    if(rescale){
        all_stats_flat <- as.data.frame(apply(all_stats_flat, 2, function(x) {return(scale(x))}))
        all_stats_flat <- t(data.frame(do.call(rbind, all_stats_flat)))
    }
    # add word column
    all_stats_flat <- cbind(all_stats_flat, all_stats_vocab)
    
    return(all_stats_flat)
}
load_all_stats <- function(){
    data_dir <- '../../data/frequency'
    out_dir <- '../../output/'
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
    return(all_stats)
}
## load change vocab
load_change_vocab <- function(){
    data_dir <- '../../data/frequency'
    word_dir <- file.path(data_dir, 'word_lists')
    growth_word_file <- file.path(word_dir, '2013_2016_growth_words_clean.csv')
    decline_word_file <- file.path(word_dir, '2013_2016_growth_decline_combined.csv')
    growth_words <- as.character(read.csv(growth_word_file, sep=',', header=TRUE)[, 'word'])
    decline_words <- as.character(read.csv(decline_word_file, header=TRUE)[, 'word'])
    change_words <- c(growth_words, decline_words)
    data <- list(growth_words=growth_words, decline_words=decline_words, change_words=change_words)
    return(data)
}
## Hirano-Imbens ADRF estimator
run_h_estimate_basic <- function(t_var, c_vars, o_var, data){
    ## Run the Hirano-Imbens estimator for average dose
    ## response, using the specified treatment, control and output.
    
    # if no control vars, we assume that every variable except
    # treatment is control
    c_vars_0 <- c_vars
    if(length(c_vars) > 0){
        c_var_str <- paste(c_vars, collapse=',')
    }
    all_vars <- Filter(function(x) { return(x != 'word')}, colnames(data))
    quant_probs <- seq(0, 0.99, by=0.01)
    if(length(c_vars_0) == 0){
        c_vars <- Filter(function(x) { return(x != t_var & x != o_var)}, all_vars)
        c_var_str <- paste(c_vars, collapse=',')
    }
    tmp_data <- as.data.frame(cbind(data[, t_var], data[, c_vars], data[, o_var]))
    colnames(tmp_data) <- c('T', c_vars, 'Y')
    tmp_data <- tmp_data[complete.cases(tmp_data[, 'Y']), ]
    grid_val <- quantile(tmp_data[,'T'], probs=quant_probs)
    # generate formulae
    treat_formula_str <- paste('T', paste(c_vars, collapse = '+'), sep = '~')
    outcome_formula_str <- paste('Y', paste(c('T', 'gps'), collapse = '+'), sep = '~')
    # convert estimator command to text and parse
    hi_est_str <- paste(c("hi_est_logit(Y=Y, treat=T, treat_formula=",treat_formula_str, ",outcome_formula=",outcome_formula_str, ",data=tmp_data, grid_val=grid_val, treat_mod='Normal', link_function='inverse')"), collapse="")
    hi_estimate <- eval(parse(text=hi_est_str))
    ## organize results
    hi_treatment_model <- hi_estimate[['t_mod']]
    hi_outcome_model <- hi_estimate[['out_mod']]
    hi_gps <- hi_estimate[['gps_fun']]
    hi_treatment_coeff <- hi_treatment_model[['coefficients']]
    hi_outcome_coeff <- hi_outcome_model[['coefficients']]
    treatment_colnames <- c('treatment_var', names(hi_treatment_coeff))
    outcome_colnames <- c('treatment_var', 'control_vars', names(hi_outcome_coeff))
    treatment_row <- c(t_var, as.vector(unlist(hi_treatment_coeff)))
    outcome_row <- c(t_var, c_var_str, as.vector(unlist(hi_outcome_coeff)))
    treatment_list <- setNames(as.list(treatment_row), treatment_colnames)
    outcome_list <- setNames(as.list(outcome_row), outcome_colnames)
    results <- list(treatment_coeff=treatment_list, outcome_coeff=outcome_list, 
                    treatment_model=hi_treatment_model,
                    outcome_model=hi_outcome_model,
                    gps=hi_gps)
    return(results)
}
# balancing data classes
balance_data <- function(data, class_var){
    class_counts <- table(data[, class_var])
    N_minority <- as.vector(sort(table(data[, class_var]), decreasing = FALSE)[1])[1]
    classes <- unique(data[, class_var])
    data_balanced <- data.frame()
    for(class_name in classes){
        relevant_data <- data[data[, class_var] == class_name,]
        N_c <- dim(relevant_data)[1]
        relevant_data <- relevant_data[sample(N_c, N_minority, replace=TRUE), ]
        if(dim(data_balanced)[1] == 0){
            data_balanced <- relevant_data
        }
        else{
            data_balanced <- rbind(data_balanced, relevant_data)
        }
    }
    return(data_balanced)
}
# Hirano-Imbens ADRF estimator over multiple treatment vars
run_h_estimate <- function(t_vars, c_var, o_var, timestep_range, stats, quant_probs=seq(0, 0.99, by=0.01)){
    ## Run the Hirano-Imbens estimator for average dose
    ## response, using the specified treatment, control and output.
    
    # if no control vars, we assume that every variable except
    # treatment is control
    c_vars_0 <- c_vars
    if(length(c_vars) > 0){
        c_var_str <- paste(c_vars, collapse=',')
    }
    all_vars <- Filter(function(x) { return(x != 'word')}, colnames(stats))
    # parameters
    param_df <- data.frame()
    param_colnames <- c('timesteps', 'control_vars', 'treatment_name', 'treatment_coeff', 'gps_coeff', 'intercept_coeff')
    # raw generated values
    vals_df <- data.frame()
    val_colnames <- c('timesteps', 'treatment_name', 'control_vars', quant_probs)
    for(t_var in t_vars){
        if(length(c_vars_0) == 0){
            c_vars <- Filter(function(x) { return(x != t_var & x != o_var)}, all_vars)
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
    # set colnames, rownames
    colnames(param_df) <- param_colnames
    rownames(param_df) <- 1:(dim(param_df)[1])
    colnames(vals_df) <- val_colnames
    rownames(vals_df) <- 1:(dim(vals_df)[1])
    # coerce quant prob values to double
    # nope this crashes things
#     for(quant_prob in quant_probs){
#         vals_df <- eval(parse(text = paste('transform(vals_df, X', quant_prob, '=as.double("X', quant_prob, '"))', sep='')))
#     }
    results <- list(params=param_df, vals=vals_df)
    return(results)
}
## causal inference: bootstrap with balanced class
run_bootstrap_balanced_class <- function(t_vars, c_vars, o_var, class_var, timestep_range, data, bootstrap_iters, quant_probs=seq(0, 0.99, by=0.01)){
    # bootstrap 
    combined_params <- data.frame()
    combined_vals <- data.frame()
    N <- dim(data)[1]
    for(i in 1:bootstrap_iters){
        # shuffle data
        data <- data[sample(N, N, replace=FALSE), ]
        tmp_data <- balance_data(data, class_var)
        results <- run_h_estimate(t_vars, c_vars, o_var, timestep_range, tmp_data, quant_probs=quant_probs)
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
    # coerce outcome probabilities to double
    treatment_level_names <- Filter(function(x){return(x != 'timesteps' & x != 'treatment_name' & x != 'control_vars' & x != 'iter')}, 
                                    colnames(combined_vals))
    combined_vals[, treatment_level_names] <- apply(combined_vals[, treatment_level_names], 2, as.double)
    # combine finally
    combined_data <- list(params=combined_params, vals=combined_vals)
    return(combined_data)
}
# training/testing
train_test_model <- function(train_data, test_data, t_var, o_var, c_vars){
    ## train/test logistic regression model to predict growth
    # (1) train
    model_results <- run_h_estimate_basic(t_var, c_vars, o_var, train_data)
    # (2) test
    # organize treatment and GPS data into common shape
    outcome_coeff_names <- c('T', 'gps', '(Intercept)')
    outcome_coef <- as.double(unlist(model_results[['outcome_coeff']][outcome_coeff_names]))
    gps_fun <- model_results[['gps']]
    gps_fun_mean <- function(x){ return(mean(gps_fun(x)))}
    gps_est <- apply(test_data[, c_vars], 1, gps_fun_mean)
    N_test <- nrow(test_data)
    test_matrix <- cbind(test_data[, t_var], gps_est, replicate(N_test, 1))
    colnames(test_matrix) <- outcome_coeff_names
    potential_outcomes <- t(outcome_coef %*% t(test_matrix))
    potential_outcomes <- 1. / (1 + exp(-potential_outcomes))
    # (3) compute accuracy => AUC
    pred_results <- prediction(potential_outcomes, test_data[, o_var])
    auc <- performance(pred_results, 'auc')
    auc <- as.numeric(auc@y.values)
    return(auc)
}
# cross-validation
k_fold_test <- function(data, t_var, o_var, c_vars, k=10){
    # compute AUC over k-fold cross-validation
    # assume that data is cleaned, balanced, etc.
    N <- nrow(data)
    data <- data[sample(N, N, replace=FALSE), ]
    k_range <- 1:k
    data_idx <- seq(1, N)
    # NOEP  we need stratified sampling plx
    folds <- createFolds(factor(data[, o_var]), k=k, list = FALSE)
#     folds <- cut(data_idx, breaks = k, labels = FALSE)
    acc <- mapply(function(i){
        test_idx <- which(folds == i, arr.ind=TRUE)
        train_idx <- which(folds != i, arr.ind=TRUE)
        train_data <- data[train_idx, ]
        test_data <- data[test_idx, ]
        auc <- train_test_model(train_data, test_data, t_var, o_var, c_vars)
        return(auc)
    }, k_range)
    return(acc)
}
# training/testing for logistic regression
train_test_LR <- function(train_data, test_data, i_vars, o_var){
    ## train/test basic logistic regression model to predict growth
    # (1) train
    model_formula <- as.formula(paste(o_var, paste(i_vars, collapse='+'), sep='~'))
    model <- glm(model_formula, data = train_data, family = binomial(link = 'logit'))
    # (2) test
    # organize treatment and GPS data into common shape
    pred_results <- predict(model, newdata = test_data, type = 'response')
    # (3) compute accuracy => AUC
    pred_results <- prediction(pred_results, test_data[, o_var])
    auc <- performance(pred_results, 'auc')
    auc <- as.numeric(auc@y.values)
    return(auc)
}
# cross-validation
k_fold_test_LR <- function(data, i_vars, o_var, k=10){
    # compute AUC over k-fold cross-validation
    # assume that data is cleaned, balanced, etc.
    N <- nrow(data)
    data <- data[sample(N, N, replace=FALSE), ]
    k_range <- 1:k
    data_idx <- seq(1, N)
    # stratified cross-validation
    folds <- createFolds(factor(data[, o_var]), k=k, list = FALSE)
#     folds <- cut(data_idx, breaks = k, labels = FALSE)
    acc <- mapply(function(i){
        test_idx <- which(folds == i, arr.ind=TRUE)
        train_idx <- which(folds != i, arr.ind=TRUE)
        train_data <- data[train_idx, ]
        test_data <- data[test_idx, ]
        auc <- train_test_LR(train_data, test_data, i_vars, o_var)
        return(auc)
    }, k_range)
    return(acc)
}
