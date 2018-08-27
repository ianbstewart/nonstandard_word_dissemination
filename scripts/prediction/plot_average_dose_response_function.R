## plot ADRF curve using bootstrapped outcome values
## from previous predictions
source('prediction_helpers.R')
set.seed(123)
## plotting code
plot_treatment_curve <- function(bootstrap_vals, t_var, treatment_level_names, x_lim=c(0,1.), y_lim=c(0,1.), title=''){
    vals_t <- bootstrap_vals[bootstrap_vals[, 'treatment_name'] == t_var, ]
    # compute confidence intervals
    vals_t <- vals_t[, treatment_level_names]
    N <- dim(vals_t)[1]
    vals_mean <- apply(vals_t, 2, mean)
    vals_sd <- apply(vals_t, 2, sd) # / N**.5
    vals_ci_upper <- vals_mean + vals_sd*1.96
    vals_ci_lower <- vals_mean - vals_sd*1.96
    # plot the mean, then the confidence intervals
    treatment_levels <- as.vector(mapply(function(x){ return(as.double(gsub("X", "", x)))}, treatment_level_names))
    # optional xlim and ylim
    if(length(x_lim) != 2){
        x_lim <- c(min(treatment_levels), max(treatment_levels))
    }
    if(length(y_lim) != 2){
        y_lim <- c(min(vals_ci_lower), max(vals_ci_upper))
    }
    x_tick_count <- 5
    x_tick_space <- x_lim[2] / x_tick_count
    x_ticks <- seq(from=x_lim[1], to=x_lim[2], by=x_tick_space)
    title_size <- 1.5
    title_full = parse(text=title)
    # ADRF line
    plot(treatment_levels, vals_mean, main = title_full, cex.main=title_size,
         type = 'l', col = 'black', 
         xlab = 'Treatment quantile', ylab = '', 
         xlim=x_lim, ylim=y_lim, 
         xaxs='i', yaxs='i', # reducing white space
         xaxt = 'n', yaxt = 'n') # custom tick marks
    # CI lines
    lines(treatment_levels, vals_ci_lower, lty = 'dashed', col = 'red')
    lines(treatment_levels, vals_ci_upper, lty = 'dashed', col = 'red')
    # chance probability line
    chance_prob <- 0.5
    lines(treatment_levels, rep(chance_prob, length(treatment_levels)), 
          col = 'black', lty='dotted')
    
    # x ticks
    axis(side=1, at=x_ticks, las=2, hadj=0.9)
}

## actual code
args <- commandArgs(trailingOnly=T)
if(length(args) == 0){
    out_dir <- '../../output/'
} else{
    out_dir <- args[1]
}

## load results from file if possible!
estimated_val_file <- file.path(out_dir, 'hi_est_all_control_success_change_outcome_results.tsv')
bootstrap_vals <- read.table(estimated_val_file, sep='\t', check.names = FALSE, header=TRUE)
t_vars <- unique(bootstrap_vals[, 'treatment_name'])
o_var <- 'success'
bootstrap_val_col_names <- colnames(bootstrap_vals)
treatment_level_names <- Filter(function(x){ return(x != 'timesteps' & x != 'treatment_name' & x != 'control_vars' & x != 'iter')}, bootstrap_val_col_names)

## actually plot lol
rows <- 1
cols <- length(t_vars)
o_var <- 'success'
treatment_levels <- as.double(treatment_level_names)
x_lim <- c(0., 1.)
y_lim <- c(0., 1.)
# set up file
# out_file <- file.path(out_dir, paste('ADRF_curves_', timestep_range, '_', quant_prob_intervals, '_', paste(t_vars, collapse=','), '.pdf', sep=''))
out_file <- gsub('.tsv', '.pdf', estimated_val_file)
width <- 10
height <- 2.5
pdf(file = out_file, width=width, height=height)
axis_label_size <- 1.75
# set up layout
# bottom, left, top, right
outer_margins <- c(5.0, 5.0, 1.0, 0.5)
inner_margins <- c(0.25, 0.75, 2.0, 0.5)
par(mfrow=c(rows, cols), oma=outer_margins, mar=inner_margins, cex.lab=axis_label_size)
ctr <- 0
y_tick_count <- 5
y_tick_space <- y_lim[2] / y_tick_count
y_ticks <- seq(from=y_lim[1], to=y_lim[2], by=y_tick_space)
t_var_titles = list(DL='D^{L}', DU='D^{U}', DS='D^{S}', DT='D^{T}')
for(t_var in t_vars){
    title <- t_var_titles[[t_var]]
    plot_treatment_curve(bootstrap_vals, t_var, treatment_level_names, 
                         x_lim=x_lim, y_lim=y_lim, title=title)
    # add just one shared y-axis for readability
    if(ctr == 0){
        axis(side=2, at=y_ticks, las=2, hadj=0.9)
        title(ylab='P(growth)', outer=TRUE)
    } else{
        # empty axis
        axis(side=2, at=y_ticks, labels=replicate(y_tick_count+1, ''), las=2, hadj=0.9)
    }
    ctr <- ctr + 1
}
title(xlab='Treatment quantile', outer=TRUE)

dev.off()