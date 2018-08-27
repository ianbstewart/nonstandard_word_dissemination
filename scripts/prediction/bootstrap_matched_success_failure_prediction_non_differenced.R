# for each bootstrap iteration: match words, use each word's covariate vector as predictor (y=1=success, y=0=failure) then predict success/failure
library('MatchIt')
library(data.table)
library('optmatch')
library('caret')
library(stringr)

## PREAMBLE: HELPER METHODS
get_dist <- function(w1, w2, M, timesteps) {
    return((sum((M[w1, timesteps] - M[w2, timesteps])**2.))**.5)
}
match_optimal <- function(M, s, k, method = 'mahalanobis') {
    fail_count <- dim(M[M[, 's'] == s, ])[1]
    M[, 'treatment'] <- 1 - M[, 'success']
    timesteps <- paste('t', (s-k-1):(s-1), sep='_')
    match_formula <- as.formula(paste('treatment', paste(timesteps, collapse = '+'), sep = ' ~ '))
    # compute distances and match
    distances <- match_on(match_formula, data=M, method=method)
    # if only one treatment variable, need to duplicate for some reason
    if(fail_count == 1){
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
get_match_sample_optimal <- function(M, failure_words, success_words, k, sample_size, method = 'mahalanobis', verbose = TRUE) {
    # Sample the failure words with replacement, match each failure word f to a success word 
    # using the optimal matching method with data between split point s and k months prior.
      # M = "match matrix"
      # failure_words = "vector of failure words"
      # success_words = "vector of success words"
      # k = "months of data prior to split point to use"
      # sample_size = "size of failure word sample"
      # method = "mahalanobis" # distance metric for optimal matching
    failure_sample <- sample(failure_words, sample_size, replace = TRUE)
    combined_words <- c(failure_sample, success_words)
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

prediction_test <- function(data, folds=0){
    # Predict success/failure with X/Y data.
      # data = "data frame with differenced covariates and success/failure output Y"
    X_cols <- colnames(data)
    X_cols <- X_cols[X_cols != 'Y']
    
    # full model fit
    formula <- as.formula(paste("Y ~ ", paste(X_cols, collapse='+'), sep=''))
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

## ACTUAL START

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
out_dir <- args[2]
bootstrap_iters <- as.integer(args[3])
k <- as.integer(args[4])

## load success/failure words and split points
success_words <- as.character(read.csv(file.path(data_dir, 'word_lists/2013_2016_growth_words_clean.csv'), header = TRUE)[, 'word'])
failure_piecewise <- as.character(read.csv(file.path(data_dir, 'word_lists/2013_2016_piecewise_growth_decline_words.csv'), header = TRUE)[, 'word'])
failure_logistic <- as.character(read.csv(file.path(data_dir, 'word_lists/2013_2016_logistic_growth_decline_words.csv'), header = TRUE)[, 'word'])
failure_logistic <- setdiff(failure_logistic, failure_piecewise)
failure_words <- c(failure_piecewise, failure_logistic)
combined_words <- c(failure_words, success_words)
split_points_piecewise <- read.table(file.path(data_dir, '2013_2016_tf_norm_2_piecewise.tsv'), sep = '\t',
                                     header = TRUE, row.names = 1)
split_points_logistic <- read.table(file.path(data_dir, '2013_2016_tf_norm_logistic_params.tsv'), sep = '\t',
                                    header = TRUE, row.names = 1)
split_points <- ceiling(c(split_points_piecewise[failure_piecewise, "x0"], split_points_logistic[failure_logistic, "loc"]))
split_points <- c(split_points, rep(0, length(success_words)))
split_points <- as.data.frame(split_points, row.names = combined_words)
colnames(split_points) <- c("s")

## build match stat matrix
tf <- read.table(file.path(data_dir, '2013_2016_tf_norm_log.tsv'), sep='\t', header = TRUE, row.names = 1)
match_stat <- tf[combined_words, ]
T <- dim(match_stat)[2]
time_range <- 0:(T-1)
time_steps <- paste('t_', time_range, sep='')
colnames(match_stat) <- time_steps
M <- cbind(match_stat, split_points)
# add success/no (counts as "treatment" in matching)
success <- as.integer(M$s == 0.0)
M <- cbind(M, success)

## load covariates
C3 <- read.table(file.path(data_dir, '2013_2016_3gram_residuals.tsv'), sep = '\t', quote = "", row.names = 1, header = TRUE)
D_U <- read.table(file.path(data_dir, '2013_2016_user_diffusion.tsv'), sep = '\t', quote = "", row.names = 1, header = TRUE)
D_S <- read.table(file.path(data_dir, '2013_2016_subreddit_diffusion.tsv'), sep = '\t', quote = "", row.names = 1, header = TRUE)
D_T <- read.table(file.path(data_dir, '2013_2016_thread_diffusion.tsv'), sep = '\t', quote = "", row.names = 1, header = TRUE)
D_U[is.na(D_U)] <- 0.
D_U[is.na(D_U)] <- 0.
D_U[is.na(D_U)] <- 0.

## organize data by feature set
feature_sets <- list(list(f=tf), 
                     list(f=tf, C3=C3), 
                     list(f=tf, D_U=D_U, D_S=D_S, D_T=D_T), 
                     list(f=tf, C3=C3, D_U=D_U, D_S=D_S, D_T=D_T)
                 )
feature_set_names <- c('f', 'f+C', 'f+D', 'f+C+D')
feature_set_count <- length(feature_set_names)

# set up prediction parameters
verbose <- FALSE
match_method <- 'mahalanobis'
FW <- length(failure_words)
sample_size <- FW
# leave two out
# folds <- sample_size
folds <- 10

## collect (1) accuracy (2) coefficients
feature_set_accuracy <- data.frame()
feature_set_coefficients <- data.frame()
iter_list <- 1:bootstrap_iters
for(f in 1:feature_set_count){
    covariates <- feature_sets[[f]]
    covariate_names <- names(covariates)
    covariate_count <- length(covariates)
    print(paste('testing covariates', paste(covariate_names, collapse=',')))
    for(i in 1:length(covariates)) {
        covariate <- covariates[[i]]
        covariate <- covariate[, sort(colnames(covariate))]
        colnames(covariate) <- time_steps
        covariates[[i]] <- covariate
    }
    feature_set_name <- feature_set_names[f]
    prediction_coefficients <- data.frame()
    accuracy_results <- data.frame()
    for(b in iter_list){
        print(paste('starting bootstrap iteration ', b, sep=''))
        match_matrix <- get_match_sample_optimal(M, failure_words, success_words, k, sample_size, method = match_method, verbose = FALSE)
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

## write results to file
accuracy_out_file <- paste(out_dir, paste('bootstrap_matched_success_failure_k', k, '_non_differenced_accuracy.tsv', sep=''), sep='/')
coefficient_out_file <- paste(out_dir, paste('bootstrap_matched_success_failure_k', k, '_non_differenced_coefficients.tsv', sep=''), sep='/')
write.table(feature_set_accuracy, file=accuracy_out_file, sep='\t', row.names = TRUE, col.names = TRUE)
write.table(feature_set_coefficients, file=coefficient_out_file, sep='\t', row.names = TRUE, col.names = TRUE)