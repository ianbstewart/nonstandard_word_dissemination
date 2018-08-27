# generate example matches for demonstration
set.seed(123)
source('prediction_helpers.R')

args <- commandArgs(trailingOnly = TRUE)
match_stat_file <- args[1]
k <- as.integer(args[2])
out_dir <- args[3]
data_dir <- args[4]

# load words
word_lists <- load_word_lists(data_dir)
success_words <- word_lists$success
failure_words <- word_lists$failure
split_points <- word_lists$split
combined_words <- c(failure_words, success_words)

# build match matrix
match_stat <- read.table(match_stat_file, sep='\t', row.names=1, header=TRUE, check.names=FALSE, quote='')
M <- get_match_stat(match_stat, combined_words, split_points)

# match stuff
sample_size <- length(failure_words)
match_method <- 'mahalanobis'
match_words <- get_match_sample_optimal(M, failure_words, success_words, k, sample_size, method=match_method, verbose=FALSE)

# write to file
out_file <- file.path(out_dir, paste('match_k', k, '_examples.tsv', sep=''))
write.table(match_words, file=out_file, sep='\t', row.names=FALSE, quote=FALSE)