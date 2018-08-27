# Match growth words with decline words
# based on frequency up until approximate split points.
library(MatchIt)
library(optmatch)

# some data processing and matching functions

smooth_stats <- function(stats) {
    smoothed <- stats + stats[stats > 0.]
    return(smoothed)
}

match_words <- function(M, k, method = 'optimal', distance = 'logit', ratio = 1., verbose = FALSE) {
    timesteps <- paste("t", as.character(0:k+1), sep="_")
    relevant_cols <- c(timesteps, "treatment")
    stats_relevant <- M[, relevant_cols]
    # need to flip treatment variable because of how optimal matching works ;_;
    if(method == 'optimal') {
        stats_relevant$treatment <- 1 - stats_relevant$treatment
    }
    match_formula <- paste("treatment", paste(timesteps, collapse=' + '), sep=' ~ ')
    match_formula <- as.formula(match_formula)
    m.out <- matchit(match_formula, data=stats_relevant, distance = distance, method=method, ratio = ratio)
    return(m.out)
}

match_all_words <- function(M, k, method = 'optimal', distance = 'logit', ratio = 1., verbose = FALSE){
    if(method == 'optimal') {
        col_names <- c("control", "treatment")
    }
    else {
        col_names <- c("treatment", "control")
    }
    m.out <- match_words(M, k, method=method, distance=distance, ratio=ratio, verbose=verbose)
    # output QQ plots to file for later ~inspekktion~
    out_dir <- '../../data/frequency/'
    out_name <- '2013_2016_growth_decline_optimal_matches.png'
    out_file <- paste(out_dir, out_name, sep='')
    png(filename=out_file)
    plot(m.out)
    dev.off()
    # reorganize match matrix
    match_matrix <- m.out$match.matrix
    match_matrix <- match_matrix[, 1]
    matches <- match_matrix[complete.cases(match_matrix)]
    matches <- cbind(names(matches), matches)
    rownames(matches) <- 1:nrow(matches)
    colnames(matches) <- col_names
    # sort by treatment order
    matches <- matches[order(matches[, "treatment"]), ]
    return(matches)
}


# prepare treatment/control units
# load growth and growth/decline words
growth_words <- as.character(read.csv('../../data/frequency/word_lists/2013_2016_growth_words_clean.csv', header = TRUE)[, 'word'])
decline_words <- as.character(read.csv('../../data/frequency/word_lists/2013_2016_decline_words.csv', header = TRUE)[, 'word'])
combined_words <- c(decline_words, growth_words)

# build match matrix
# load match stat from file
match_stat <- read.table('../../data/frequency/2013_2016_tf_norm.tsv', sep = '\t', header = TRUE, row.names = 1)
match_stat[is.na(match_stat)] <- 0
match_stat <- log10(smooth_stats(match_stat))
match_stat <- match_stat[combined_words, ]
timesteps <- paste("t", 1:ncol(match_stat), sep='_')
colnames(match_stat) <- timesteps
print(head(match_stat))

# add treatment condition
C <- length(decline_words)
T <- length(growth_words)
treatment <- c(rep(0, C), rep(1, T))
M <- cbind(match_stat, treatment)
print(head(M))

# matching
# now match treatment and control words
method <- 'optimal'
distance <- 'logit'
ratio <- 1.
verbose <- TRUE
# matching on first timestep
k <- 1
matches <- match_all_words(M, k, method=method, distance=distance, ratio=ratio, verbose=verbose)
if(verbose) {
	    print(head(matches))
}

# write
# write to file
out_dir <- '../../data/frequency'
out_file <- paste(out_dir, paste('2013_2016_growth_decline_', method, '_matches.tsv', sep=''), sep='/')
write.table(matches, file=out_file, sep='\t')