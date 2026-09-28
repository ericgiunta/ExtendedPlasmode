library(Colossus)
library(data.table)
library(parallel)

options("width"=200)

df <- fread("NEW_RUNS/df_total.csv")

run_dose_model <- function(df, outcome, dose){
    base_model <- as.formula(paste("Cox(age_entry, age_exit, ",outcome,") ~ loglinear(factor(yob_cat10), factor(SES_HighestEd), sexm) + M()",sep=""))
    hr_model <- as.formula(paste("Cox(age_entry, age_exit, ",outcome,") ~ loglinear(", dose, ", factor(yob_cat10), factor(SES_HighestEd), sexm) + M()",sep=""))
    err_model <- as.formula(paste("Cox(age_entry, age_exit, ",outcome,") ~ plinear(", dose, ") + loglinear(factor(yob_cat10), factor(SES_HighestEd), sexm) + M()",sep=""))
    control <- list(Ncores=16, maxiters=c(10, 10), epsilon = 1e-6, der_epsilon = 1e-6, 'verbose'=F)
    e_base <- CoxRun(base_model, df, control = control)
    a_n <- list(c(0.001, e_base$beta_0), c(0.0001, e_base$beta_0), c(0.01, e_base$beta_0))
    control <- list(Ncores=16, maxiters=c(20, 50), epsilon = 1e-6, der_epsilon = 1e-6, 'verbose'=F)
    e0 <- CoxRun(hr_model, df, control = control, a_n = a_n, norm = 'max')
    e1 <- CoxRun(err_model, df, control = control, a_n = a_n, norm = 'max')
    #
    print(outcome)
    print(e0)
    print(e1)
    model_type <- "null"
    a_n <- c(0)
    if (e0$LogLik < e1$LogLik){
      model_type <- "err"
      a_n <- e1$beta_0
      model <- as.formula(paste("Cox(age_entry, age_exit, ",outcome,") ~ plinear(", dose, ") + loglinear(factor(yob_cat10), factor(SES_HighestEd), sexm) + M()",sep=""))
    } else {
      model_type <- "hr"
      a_n <- e0$beta_0
      model <- as.formula(paste("Cox(age_entry, age_exit, ",outcome,") ~ loglinear(", dose, ", factor(yob_cat10), factor(SES_HighestEd), sexm) + M()",sep=""))
    }
    print(model_type)
    control <- list(Ncores=16, maxiters=c(1, 1), epsilon = 1e-6, der_epsilon = 1e-6, 'verbose'=F)
    e0 <- CoxRun(model, df, control = control, a_n = a_n)
    plot_options <- list(
      "type" = c("surv", "../PLOTS/lung_surv"), "studyid" = "StudyID",
      "verbose" = 0, "surv_curv" = T, "martingale" = F, "strat_haz" = F, "km" = F
      )
    e1 <- plot(e0, df, plot_options = plot_options)
    norm_surv <- e1[["standard"]]
    fwrite(norm_surv, file = paste("NEW_RUNS/", outcome, "_Background.csv", sep = ""))
}


run_nodose_model <- function(df, outcome){
    hr_model <- as.formula(paste("Cox_Strata(age_entry, age_exit, ",outcome,", yob_cat10) ~ loglinear(factor(SES_HighestEd), sexm) + M()",sep=""))
    control <- list(Ncores=16, maxiters=c(50, 50), epsilon = 1e-6, der_epsilon = 1e-6, 'verbose'=F)
    e0 <- CoxRun(hr_model, df, control = control)
    #
    print(outcome)
    print(e0)
    a_n <- e0$beta_0
    model <- as.formula(paste("Cox_Strata(age_entry, age_exit, ",outcome,", yob_cat10) ~ loglinear(factor(SES_HighestEd), sexm) + M()",sep=""))
    control <- list(Ncores=16, maxiters=c(1, 1), epsilon = 1e-6, der_epsilon = 1e-6, 'verbose'=F)
    e0 <- CoxRun(hr_model, df, control = control, a_n = a_n)
    plot_options <- list(
      "type" = c("surv", "../PLOTS/lung_surv"), "studyid" = "StudyID",
      "verbose" = 0, "surv_curv" = T, "martingale" = F, "strat_haz" = T, "km" = F
      )
    e1 <- plot(e0, df, plot_options = plot_options)
    norm_surv <- e1[["standard"]]
    fwrite(e1, file = paste("NEW_RUNS/", outcome, "_Background.csv", sep = ""))
}


out_col <- c('Esophagus', 'lung', 'Parkinsons', 'Ischemic', 'CLL', 'AML', 'NonCLL_excAML', "Heart_excIsch", "AllLymphHema_excLeukAleuk", "AllCancer_excElse", "allevent", "censored")
# Esophagus
##sink("NEW_RUNS/eso.txt")
outcomes <- c("Esophagus")
dose <- "cumulative_dose_esophagus_lag10"
for (outcome in outcomes){
    run_dose_model(df, outcome, dose)
}
#stop()
#sink()
# Lung
#sink("NEW_RUNS/lung.txt")
outcomes <- c("lung")
dose <- "cumulative_dose_lung_lag10"
for (outcome in outcomes){
    run_dose_model(df, outcome, dose)
}
# Brain
outcomes <- c("Parkinsons")
dose <- "cumulative_dose_brain_lag10"
for (outcome in outcomes){
    run_dose_model(df, outcome, dose)
}
#sink()
# Heart
#sink("NEW_RUNS/heart.txt")
outcomes <- c("Ischemic", "Heart_excIsch")
dose <- "cumulative_dose_heart_lag10"
for (outcome in outcomes){
    run_dose_model(df, outcome, dose)
}
#sink()
# Colon
#sink("NEW_RUNS/colon.txt")
outcomes <- c("CLL", "AML", "NonCLL_excAML", "AllLymphHema_excLeukAleuk")
dose <- "cumulative_dose_colon_lag10"
for (outcome in outcomes){
    run_dose_model(df, outcome, dose)
}
#sink()
# All
#sink("NEW_RUNS/all.txt")
outcomes <- c("AllCancer_excElse")#, "allevent", "censored")
dose <- "cumulative_dose_colon_lag10"
for (outcome in outcomes){
    run_dose_model(df, outcome, dose)
}
#sink()
# None
#sink("NEW_RUNS/none.txt")
outcomes <- c( "censored")
for (outcome in outcomes){
    run_nodose_model(df, outcome)
}
#sink()
