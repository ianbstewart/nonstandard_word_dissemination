# make bar plot of success/failure prediction accuracy
library('ggplot2')

args <- commandArgs(trailingOnly = TRUE)
accuracy_file <- args[1]
out_dir <- args[2]
file_base <- basename(accuracy_file)
out_file <- file.path(out_dir, gsub('.tsv', '.png', file_base))
print(out_file)

# load data, aggregate
print(accuracy_file)
accuracy_results <- read.table(accuracy_file, sep='\t', header = TRUE, check.names = FALSE)
print(head(accuracy_results))
accuracy_results[, 'feature_set_name'] <- as.factor(accuracy_results[, 'feature_set_name'])
folds <- unique(accuracy_results[, 'folds'])[1]
accuracy_results[, 'AccuracySD'] <- accuracy_results[, 'AccuracySD'] * 100 / (folds)**.5
accuracy_results[, 'Accuracy'] <- accuracy_results[, 'Accuracy'] * 100

accuracy_aggregate <- aggregate(as.formula('. ~ feature_set_name'), accuracy_results, mean)
# sort by accuracy
accuracy_aggregate <- accuracy_aggregate[with(accuracy_aggregate, order(Accuracy)), ]

## plot!!
barplot_margins = c(5, 5, 5, 5)
par(mar = barplot_margins)
bar_x <- 1:(dim(accuracy_aggregate)[1])
bar_y <- accuracy_aggregate[, 'Accuracy']
err_y <- accuracy_aggregate[, 'AccuracySD']
feature_set_names <- accuracy_aggregate[, 'feature_set_name']
ylabel <- 'Accuracy'
# ylim <- c(min(bar_y) - max(err_y)*2, max(bar_y) + max(err_y)*2)
ylim <- c(0, max(bar_y) + max(err_y)*4)
png(filename=out_file)
bars <- barplot(height = bar_y, ylim=ylim, names.arg = feature_set_names, ylab = ylabel)
err_bars <- segments(bars, bar_y - err_y, bars, bar_y + err_y, lwd = 1.5)
err_arrows <- arrows(bars, bar_y - err_y, bars, bar_y + err_y, lwd = 1.5, angle = 90, code = 3, length = 0.05)

# add annotations with mean values
max_digits <- 4
annotate_y_offset <- 2.
annotate_cex <- 2.0
mean_str <- lapply(bar_y, function(x){return(format(x, digits=max_digits))})
annotate_y <- bar_y + annotate_y_offset
text(bars, y=annotate_y, labels=mean_str, cex=annotate_cex)
dev.off()