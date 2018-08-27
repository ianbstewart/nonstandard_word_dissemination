# match each growth-decline word with a growth word 
# based on similar frequency between timestep s-k and s
# where s is the splitpoint for the growth-decline word
library('MatchIt')
library(data.table)
library('optmatch')
# load data
smooth_stats <- function(stats) {
    smoothed <- stats + stats[stats > 0.]
    return(smoothed)
}
args <- commandArgs(trailingOnly=T)
data_dir <- args[1]
out_dir <- args[2]
replace <- as.logical(args[3])

growth_words <- as.character(read.csv(paste(data_dir, 'word_lists/2013_2016_growth_words_clean.csv',sep='/'), header = TRUE)[, 'word'])
growth_decline_piecewise <- as.character(read.csv(paste(data_dir, 'word_lists/2013_2016_piecewise_growth_decline_words.csv', sep='/'), header = TRUE)[, 'word'])
growth_decline_logistic <- as.character(read.csv(paste(data_dir, 'word_lists/2013_2016_logistic_growth_decline_words.csv', sep='/'), header = TRUE)[, 'word'])
growth_decline_logistic <- setdiff(growth_decline_logistic, growth_decline_piecewise)
growth_decline_words <- c(growth_decline_piecewise, growth_decline_logistic)
combined_words <- c(growth_decline_words, growth_words)
# collect split points!! for all words
split_points_piecewise <- read.table(paste(data_dir, '2013_2016_tf_norm_2_piecewise.tsv', sep='/'), sep = '\t', 
                                     header = TRUE, row.names = 1)
split_points_logistic <- read.table(paste(data_dir, '2013_2016_tf_norm_logistic_params.tsv', sep='/'), sep = '\t',
                                    header = TRUE, row.names = 1)
split_points <- ceiling(c(split_points_piecewise[growth_decline_piecewise, "x0"], split_points_logistic[growth_decline_logistic, "loc"]))
split_points <- c(split_points, rep(0, length(growth_words)))
print('reformatting split points')
split_points <- as.data.frame(split_points, row.names = combined_words)
colnames(split_points) <- c("s")

print('loading match stat')
match_stat <- read.table(paste(data_dir, '2013_2016_tf_norm.tsv', sep='/'), sep = '\t', header = TRUE, row.names = 1, quote="")
match_stat[is.na(match_stat)] <- 0.
match_stat <- log10(smooth_stats(match_stat))
match_stat <- match_stat[combined_words,]

# build match matrix
N <- dim(match_stat)[2]
time_range <- 0:(N-1)
time_steps <- paste('t_', time_range, sep='')
colnames(match_stat) <- time_steps
M <- cbind(match_stat, split_points)
# add treatment (growth word is treatment for sake of matching...yeah it suxx)
treatment <- as.integer(M$s == 0.0)
M <- cbind(M, treatment)

# match all the words on all combinations of split point s and k
match_words <- function(M, s, k, treatment_stats, method = 'optimal', distance = 'logit', ratio = 1.) {
    timesteps <- paste("t", as.character((s-k-1):(s-1)), sep="_")
    relevant_cols <- c(timesteps, "treatment")
    control_relevant <- M[M$s == s, relevant_cols]
    treatment_relevant <- treatment_stats[, relevant_cols]
    stats_relevant <- rbind(control_relevant, treatment_relevant)
    # need to flip treatment variable because of how optimal matching works ;_;
    if(method == 'optimal') {
        stats_relevant$treatment <- 1 - stats_relevant$treatment
    }
    C <- nrow(control_relevant)
    print(paste(C, "relevant control vars"), sep = '')
    match_formula <- paste("treatment", paste(timesteps, collapse=' + '), sep=' ~ ')
    match_formula <- as.formula(match_formula)
    m.out <- matchit(match_formula, data=stats_relevant, distance = distance, method=method, ratio = ratio)
    return(m.out)
}

unique_split_points <- sort(unique(M[M$s > 0, ]$s))
S_min <- 3
S_max <- max(unique_split_points)
K_min <- 0
K_max <- 8
k_range <- K_min:K_max
treatment_stats <- M[M$s == 0, ]
global_matches <- data.table()
# all_m_out <- list()
match_cols <- c('word', 'match', 's', 'k')
method <- 'optimal'
for(k in k_range){ 
#     all_m_out[[k]] <- list()
    split_points_k <- unique_split_points[unique_split_points > k+1]
    for(s in split_points_k){
        m.out <- match_words(M, s, k, treatment_stats, method=method)
        matches <- cbind(rownames(m.out$match.matrix), m.out$match.matrix[, 1])
        matches <- matches[!is.na(matches[, 2]), ]
        # if only 1 match, need to restructure
        if(length(matches) == 2){
            matches <- t(data.table(matches))
        }
        # for each row, add word,match,s,k
        matches <- cbind(names(matches), matches, s, k)
        colnames(matches) <- match_cols
        global_matches <- rbind(global_matches, matches)
	# if we're not replacing, then remove from treatment stats
	if(! replace){
	     
	}
#         all_m_out[[k]][[s]] <- m.out
    }
}
print(head(global_matches))
N <- dim(global_matches)[1]
global_matches_table <- data.frame(matrix(unlist(global_matches), nrow=N, byrow=F),stringsAsFactors=FALSE)
colnames(global_matches_table) <- colnames(global_matches)
global_matches_table[, "s"] <- as.integer(global_matches_table[, "s"])
global_matches_table[, "k"] <- as.integer(global_matches_table[, "k"])
print('got global matches table')
print(head(global_matches_table))

# compute match differences
get_diff <- function(r) {
    word <- r[1]
    match <- r[2]
    s <- as.integer(r[3])
    k <- as.integer(r[4])
    relevant_dates <- s-k:s
    diff <- sum(abs(match_stat[word, relevant_dates] - match_stat[match, relevant_dates])) 
    return(diff)
}
diffs <- apply(global_matches_table, 1, get_diff)
print(paste('got diffs', head(diffs)))
global_matches_table <- cbind(global_matches_table, diffs)
colnames(global_matches_table)[5] <- 'diff'
print('now with diffs')
print(head(global_matches_table))

# write to file

out_file <- paste(out_dir, 'growth_growth_decline_optimal_matchit_results.tsv', sep='/')
write.table(global_matches_table, sep='\t', file = out_file, quote = F, row.names = F)