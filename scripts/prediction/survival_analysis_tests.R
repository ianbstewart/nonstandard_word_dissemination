# compute deviance for different Cox regression models
# with ANOVA between model and null model
library(survival)
source('prediction_helpers.R')

args <- commandArgs(trailingOnly=T)
if(length(args) == 0){
    out_dir <- '../../output/'
} else{
    out_dir <- args[1]
}
data_dir <- '../../data/frequency'
# out_dir <- '../../output'

# load data	
tf <- read.table(file.path(data_dir, '2013_2016_tf_norm_log.tsv'), sep = '\t', header = TRUE, row.names = 1)
D_L <- read.table(file.path(data_dir, '2013_2016_3gram_residuals.tsv'), sep = '\t', quote = "", header = TRUE, row.names = 1)
D_U <- read.table(file.path(data_dir, '2013_2016_user_diffusion_log.tsv'), sep = '\t', quote = "", header = TRUE, row.names = 1)
D_S <- read.table(file.path(data_dir, '2013_2016_subreddit_diffusion_log.tsv'), sep = '\t', quote = "", header = TRUE, row.names = 1)
D_T <- read.table(file.path(data_dir, '2013_2016_thread_diffusion_log.tsv'), sep = '\t', quote = "", header = TRUE, row.names = 1)
word_dir <- file.path(data_dir, 'word_lists')
# load word lists
word_lists <- load_word_lists()
success_words <- word_lists[['success']]
fail_words <- word_lists[['failure']]
split_points <- word_lists[['split']][, 's']
print(paste(length(success_words), ' success words', sep=''))
print(paste(length(fail_words), ' fail words', sep=''))
print(paste(length(split_points), ' split points', sep=''))
combined_words <- c(fail_words, success_words)
N <- ncol(tf)
split_points_growth <- matrix(rep(N, length(success_words)))
split_points_combined <- c(split_points, split_points_growth)

# organize survival data
# use first m months of data
m <- 3
covariate_names <- c('f', 'D_L', 'D_U', 'D_S', 'D_T')
covariate_matrix <- cbind(apply(tf[combined_words, 1:m], 1, mean), 
                          apply(D_L[combined_words, 1:m], 1, mean),
                          apply(D_U[combined_words, 1:m], 1, mean),
                          apply(D_S[combined_words, 1:m], 1, mean),
                          apply(D_T[combined_words, 1:m], 1, mean))
# fill NA values
covariate_matrix[is.na(covariate_matrix)] = 0
colnames(covariate_matrix) <- covariate_names
# rescale EVERYTHING
covariate_matrix <- scale(covariate_matrix)

# add death times
# death times, death 1/0, covariates
death_time <- split_points_combined
# T=treatment, C=control, V=...vocabulary JUST GO WITH IT
T <- length(fail_words)
C <- length(success_words)
V <- length(combined_words)
# death 1/0
death <- c(rep(1, T), rep(0, C))
# constant
const <- rep(1, V)
survival_matrix <- cbind(death_time, death, const)
# survival_matrix <- cbind(death_time, death)
# combine EVERYTHING
rownames(survival_matrix) <- combined_words
survival_matrix <- cbind(survival_matrix, covariate_matrix)
survival_matrix <- as.data.frame(survival_matrix)

# survival analysis: all factors
formula <- as.formula("Surv(time=death_time, event = death) ~ f + D_L + D_U + D_S + D_T")
cox_model <- coxph(formula, data=survival_matrix)
out_file <- file.path(out_dir, 'cox_regression_deviance_output.txt')
anova_results <- anova(cox_model)
capture.output(anova_results, file=out_file, append=F)

# survival analysis: separate factor sets
test_feature_set <- function(feature_set_formula, survival_matrix, out_file) {
    formula <- as.formula(paste("Surv(time=death_time, event = death) ~ ", feature_set_formula, sep = ''))
    cox_model <- coxph(formula, data=survival_matrix)
    model_summary <- summary(cox_model)
    print('bout to write model summary')
    # write(model_summary, file=out_file, append=T)
    capture.output(model_summary, file=out_file, append=T)
    anova_results <- anova(cox_model)
    print('bout to write ANOVA results')
    capture.output(anova_results, file=out_file, append=T)
}


feature_sets <- list( c("f"), c("f", "D_L"), c("f", "D_U", "D_S", "D_T"), c("f", "D_U", "D_S", "D_T"))
for (feature_set in feature_sets) {
    feature_set_formula <- paste(feature_set, collapse='+')
    print(paste('testing feature set ', feature_set_formula, sep=''))
    test_feature_set(feature_set_formula, survival_matrix, out_file)
}