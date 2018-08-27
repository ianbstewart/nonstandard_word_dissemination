# stolen from hi_est.R (https://github.com/cran/causaldrf/blob/master/R/hi_est.R)
hi_est_logit <-  function (Y,
                     treat,
                     treat_formula,
                     outcome_formula,
                     data,
                     grid_val,
                     treat_mod,
                     link_function,
                     ...)
{
  # Y is the name of the Y variable
  # treat is the name of the treatment variable
  # treat_formula is the formula for the treatment model
  # data will contain all the data: X, treat, and Y
  # grid_val is the set of grid points on T
  # treat_mod is th treatment model to fit
  # link_function is the link function used, if needed

  # The outcome is a list of 3 objects:
  #    (1) estimated values of ADRF at gridvalues
  #    (2) result of treatment model
  #    (3) result of the outcome model


  #save input
  tempcall <- match.call()


  #some basic input checks
  if (!("Y" %in% names(tempcall))) stop("No Y variable specified")
  if (!("treat" %in% names(tempcall)))  stop("No treat variable specified")
  if (!("treat_formula" %in% names(tempcall))) stop("No treat_formula model specified")
  if (!("outcome_formula" %in% names(tempcall))) stop("No outcome_formula model specified")
  if (!("data" %in% names(tempcall))) stop("No data specified")
  if (!("grid_val" %in% names(tempcall)))  stop("No grid_val specified")
  if (!("treat_mod" %in% names(tempcall)) | ("treat_mod" %in% names(tempcall) & !(tempcall$treat_mod %in% c("NegBinom", "Poisson", "Gamma", "LogNormal", "Sqrt", "Binomial", "Normal")))) stop("No valid family specified (\"NegBinom\", \"Poisson\", \"Gamma\", \"Log\", \"Sqrt\", \"Binomial\", \"Normal\")")
  if (tempcall$treat_mod == "Gamma") {if(!(tempcall$link_function %in% c("log", "inverse"))) stop("No valid link function specified for family = Gamma (\"log\", \"inverse\")")}
  if (tempcall$treat_mod == "Binomial") {if(!(tempcall$link_function %in% c("logit", "probit", "cauchit", "log", "cloglog"))) stop("No valid link function specified for family = Binomial (\"logit\", \"probit\", \"cauchit\", \"log\", \"cloglog\")")}
  if (tempcall$treat_mod == "Ordinal" ) {if(!(tempcall$link_function %in% c("logit", "probit", "cauchit", "cloglog"))) stop("No valid link function specified for family = Ordinal (\"logit\", \"probit\", \"cauchit\", \"cloglog\")")}

  #make new dataframe for newly computed variables, to prevent variable name conflicts
  tempdat <- data.frame(
    Y = data[,as.character(tempcall$Y)],
    treat = data[,as.character(tempcall$treat)]
  )

#-------------------------------------------------------------------------------

  # make a formula for the treatment model
  #   formula_t <- eval(parse(text = paste(deparse(tempcall$treat, width.cutoff = 500), deparse(tempcall$treat_formula, width.cutoff = 500), sep = "")))
  formula_t <- tempcall$treat_formula

  if (treat_mod == "NegBinom") {
    samp_dat <- data
    result <- MASS::glm.nb(formula = formula_t, link = log, data = samp_dat)
    cond_mean <- result$fitted.values
    cond_var <- cond_mean + cond_mean^2/result$theta
    prob_nb_est <- (cond_var - cond_mean)/cond_var
    gps_fun_NB <- function(tt) {
      dnbinom(x = tt, size = result$theta, mu = result$fitted.values,
              log = FALSE)
    }
    gps_fun <- gps_fun_NB
  }
  else if (treat_mod == "Poisson") {
    samp_dat <- data
    result <- glm(formula = formula_t, family = "poisson", data = samp_dat)
    cond_mean <- result$fitted.values
    samp_dat$gps_vals <- dpois(x = tempdat$treat, lambda = cond_mean)
    gps_fun_Pois <- function(t) {
      dpois(t, lambda = cond_mean)
    }
    gps_fun <- gps_fun_Pois
  }
  else if (treat_mod == "Gamma") {
    samp_dat <- data
    result <- glm(formula = formula_t, family = Gamma(link = link_function),
                  data = samp_dat)
    est_treat <- result$fitted
    shape_gamma <- as.numeric(MASS::gamma.shape(result)[1])
    theta_given_X <- result$fitted.values/shape_gamma
    theta_treat_X <- tempdat$treat/shape_gamma
    gps_fun_Gamma <- function(t) {
      dgamma(t, shape = shape_gamma, scale = theta_given_X)
    }
    gps_fun <- gps_fun_Gamma
  }
  else if (treat_mod == "LogNormal") {

    samp_dat <- data
    samp_dat[, as.character(tempcall$treat)] <- log(samp_dat[, as.character(tempcall$treat)])
    result <- lm(formula = formula_t, data = samp_dat)
    est_log_treat <- result$fitted
    sigma_est <- summary(result)$sigma
    gps_fun_Log <- function(tt) {
      dnorm(log(tt), mean = est_log_treat, sd = sigma_est)
    }
    gps_fun <- gps_fun_Log
  }
  else if (treat_mod == "Sqrt") {

    samp_dat <- data
    samp_dat[, as.character(tempcall$treat)] <- sqrt(samp_dat[, as.character(tempcall$treat)])
    result <- lm(formula = formula_t, data = samp_dat)
    est_sqrt_treat <- result$fitted
    sigma_est <- summary(result)$sigma
    gps_fun_sqrt <- function(tt) {
      dnorm(sqrt(tt), mean = est_sqrt_treat, sd = sigma_est)
    }
    gps_fun <- gps_fun_sqrt
  }
  else if (treat_mod == "Normal") {
    samp_dat <- data
    result <- lm(formula = formula_t, data = samp_dat)
    gps_fun_Normal <- function(tt) {
      dnorm(tt, mean = result$fitted, sd = summary(result)$sigma)
    }
    gps_fun <- gps_fun_Normal
  }

  else if (treat_mod == "Binomial") {
    samp_dat <- data

    if(tempcall$link_function == "logit") lf <- binomial(link = logit)
    if(tempcall$link_function == "probit") lf  <- binomial(link = probit)
    if(tempcall$link_function == "cauchit") lf  <- binomial(link = cauchit)
    if(tempcall$link_function == "log") lf  <- binomial(link = log)
    if(tempcall$link_function == "cloglog") lf  <- binomial(link = cloglog)


      result <- glm(formula_t, family = lf, data = samp_dat)

      samp_dat$prob_1[tempdat$treat == 0] <- 1 - predict.glm(result, type = "response")[tempdat$treat == 0]
      samp_dat$prob_1[tempdat$treat == 1] <- predict.glm(result, type = "response")[tempdat$treat == 1]

      gps_fun_Binomial <- function(tt) ifelse(tt == 1, samp_dat$prob_1, 1 - samp_dat$prob_1)
      gps_fun <- gps_fun_Binomial

  }
        
  else {
    stop("No valid treat_mod specified.  Please try again.")
  }

#---------------------------------

  # using the estimated gps, get the estimate for the ADRF using the hi method.

  tempdat$gps <- gps_fun(tempdat$treat)

  # this allows for the user to use flexible outcome models
colnames(tempdat) <- c(as.character(tempcall$Y), as.character(tempcall$treat), "gps" )

# outcome_model <- lm(outcome_formula, data = tempdat, ...)
# estimate outcome (probability) using logistic regression
outcome_model <- glm(outcome_formula, data = tempdat, family=binomial(link='logit'))

outcome_coef <- outcome_model$coef

# need to create model matrix with gps evaluated at grid values and then average over all the units.

mean_outcome_grid <- numeric(length(grid_val))
for (i in 1:length(grid_val)) {

  temp_matrix <- cbind(numeric(nrow(tempdat)), grid_val[i], gps_fun(grid_val[i]))
  colnames(temp_matrix) <- c(as.character(tempcall$Y), as.character(tempcall$treat), "gps" )
  temp_matrix <- data.frame(temp_matrix)

  # utils::str(m_frame <- model.frame(outcome_formula, tempdat))
  m_frame <- model.frame(outcome_formula, temp_matrix)
  covar_temp <- model.matrix(outcome_formula, m_frame)
  covar_grid_mat <- covar_temp

  potential_outcomes_at_t <- t(outcome_coef %*% t(covar_grid_mat))
  potential_outcomes_at_t <- 1. / (1 + exp(-potential_outcomes_at_t))
  # log-transform for logistic probabilities

  mean_outcome_grid[i] <- mean(potential_outcomes_at_t)

}

z_object <- list(param = mean_outcome_grid,
                 t_mod = result,
                 out_mod = outcome_model,
                 call = tempcall,
                 gps_fun = gps_fun)

class(z_object) <- "causaldrf"
z_object


}