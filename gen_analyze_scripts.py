import numpy as np

temp_script_single = '''library(data.table)
library(dplyr)
library(Colossus)
library(survival)
library(arrow)

options("width"=200)


compx_round <- function(x,digits){
  a <- floor(x)
  b <- x-a
  b <- b*10^digits
  b <- ceiling(b)
  b <- b*10^(-1*digits)
  a+b
}


DS <- arrow::open_dataset(sources = "../WITH_EVENTS/Subj_{count}_{name}")
## Create a scanner
SO <- Scanner$create(DS)
## Load it as n Arrow Table in memory
AT <- SO$ToTable()
## Convert it to an R data frame
df_comb <- as.data.table(AT)

df_comb <- df_comb[, age_entry:=as.numeric(age_entry)]
df_comb <- df_comb[, age_exit:=as.numeric(age_exit)]
df_comb <- df_comb[, sexm:=as.numeric(sexm)]
df_comb <- df_comb[, yob_cat10:=as.numeric(yob_cat10)]

df_comb <- df_comb[, cumulative_dose_brain_lag10:=as.numeric(cumulative_dose_brain_lag10)]
df_comb <- df_comb[, cumulative_dose_colon_lag10:=as.numeric(cumulative_dose_colon_lag10)]
df_comb <- df_comb[, cumulative_dose_esophagus_lag10:=as.numeric(cumulative_dose_esophagus_lag10)]
df_comb <- df_comb[, cumulative_dose_heart_lag10:=as.numeric(cumulative_dose_heart_lag10)]
df_comb <- df_comb[, cumulative_dose_lung_lag10:=as.numeric(cumulative_dose_lung_lag10)]

df_comb <- df_comb[, exit_adjust_AML:=as.numeric(exit_adjust_AML)]
df_comb <- df_comb[, exit_adjust_AllCancer_excElse:=as.numeric(exit_adjust_AllCancer_excElse)]
df_comb <- df_comb[, exit_adjust_AllLymphHema_excLeukAleuk:=as.numeric(exit_adjust_AllLymphHema_excLeukAleuk)]
df_comb <- df_comb[, exit_adjust_CLL:=as.numeric(exit_adjust_CLL)]
df_comb <- df_comb[, exit_adjust_Esophagus:=as.numeric(exit_adjust_Esophagus)]
df_comb <- df_comb[, exit_adjust_Heart_excIsch:=as.numeric(exit_adjust_Heart_excIsch)]
df_comb <- df_comb[, exit_adjust_Ischemic:=as.numeric(exit_adjust_Ischemic)]
df_comb <- df_comb[, exit_adjust_NonCLL_excAML:=as.numeric(exit_adjust_NonCLL_excAML)]
df_comb <- df_comb[, exit_adjust_Parkinsons:=as.numeric(exit_adjust_Parkinsons)]
df_comb <- df_comb[, exit_adjust_censor:=as.numeric(exit_adjust_censor)]
df_comb <- df_comb[, exit_adjust_lung:=as.numeric(exit_adjust_lung)]

df_comb$exit_adjust_AML                       <- round(df_comb$exit_adjust_AML, digits=3)
df_comb$exit_adjust_AllCancer_excElse         <- round(df_comb$exit_adjust_AllCancer_excElse, digits=3)
df_comb$exit_adjust_AllLymphHema_excLeukAleuk <- round(df_comb$exit_adjust_AllLymphHema_excLeukAleuk, digits=3)
df_comb$exit_adjust_CLL                       <- round(df_comb$exit_adjust_CLL, digits=3)
df_comb$exit_adjust_Esophagus                 <- round(df_comb$exit_adjust_Esophagus, digits=3)
df_comb$exit_adjust_Heart_excIsch             <- round(df_comb$exit_adjust_Heart_excIsch, digits=3)
df_comb$exit_adjust_Ischemic                  <- round(df_comb$exit_adjust_Ischemic, digits=3)
df_comb$exit_adjust_NonCLL_excAML             <- round(df_comb$exit_adjust_NonCLL_excAML, digits=3)
df_comb$exit_adjust_Parkinsons                <- round(df_comb$exit_adjust_Parkinsons, digits=3)
df_comb$exit_adjust_censor                    <- round(df_comb$exit_adjust_censor, digits=3)
df_comb$exit_adjust_lung                      <- round(df_comb$exit_adjust_lung, digits=3)



#
df_comb <- df_comb[, event_t1 := fifelse(event_{outcome} == 1, exit_adjust_{outcome}, 1000)]
df_comb <- df_comb[, event_t  := fifelse(event_t1 < 1000, event_t1, 1000)]

StudyID_first_event <- df_comb %>% group_by(StudyID) %>% summarise(event_age = min(event_t))
df_comb <- df_comb[StudyID_first_event, on = c("StudyID"), nomatch = 0]
df_comb <- df_comb[age_entry < event_age,]

df_comb <- df_comb[, exit_adjust := fifelse(event_{outcome} == 1, exit_adjust_{outcome}, age_exit)]
df_comb$event_1 <- as.integer(df_comb$event_{outcome} == 1)
df_comb$exit_adjust<- round(df_comb$exit_adjust, digits=2)



print("---------------------------------COLOSSUS START---------------------------------------------")
model <- Cox(age_entry, exit_adjust, event_1) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm)
control <- list("Ncores"=16, maxiters=c(1,20), 'verbose'=F)
#ecox0 <- CoxRun(model, df_comb, control = control, norm = "max")
model <- Cox(age_entry, exit_adjust, event_1) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm) + plinear({dose})
a_n <- c(-0.3, -0.1, -0.6, -0.3, 0.001, -0.01, -0.03, 0.03, -0.4, -0.2, -0.2, -1.2, 2)
ecox1 <- CoxRun(model, df_comb, control = control, a_n = a_n, norm = "max")

print("---------------------------------COX REGRESSION RESULT---------------------------------")
#print(ecox0)
print(ecox1)
###
'''

temp_script_group_heart = '''library(data.table)
library(dplyr)
library(Colossus)
library(survival)
library(arrow)

options("width"=200)


compx_round <- function(x,digits){
  a <- floor(x)
  b <- x-a
  b <- b*10^digits
  b <- ceiling(b)
  b <- b*10^(-1*digits)
  a+b
}


DS <- arrow::open_dataset(sources = "../WITH_EVENTS/Subj_{count}_{name}")
## Create a scanner
SO <- Scanner$create(DS)
## Load it as n Arrow Table in memory
AT <- SO$ToTable()
## Convert it to an R data frame
df_comb <- as.data.table(AT)

df_comb <- df_comb[, age_entry:=as.numeric(age_entry)]
df_comb <- df_comb[, age_exit:=as.numeric(age_exit)]
df_comb <- df_comb[, sexm:=as.numeric(sexm)]
df_comb <- df_comb[, yob_cat10:=as.numeric(yob_cat10)]

df_comb <- df_comb[, cumulative_dose_brain_lag10:=as.numeric(cumulative_dose_brain_lag10)]
df_comb <- df_comb[, cumulative_dose_colon_lag10:=as.numeric(cumulative_dose_colon_lag10)]
df_comb <- df_comb[, cumulative_dose_esophagus_lag10:=as.numeric(cumulative_dose_esophagus_lag10)]
df_comb <- df_comb[, cumulative_dose_heart_lag10:=as.numeric(cumulative_dose_heart_lag10)]
df_comb <- df_comb[, cumulative_dose_lung_lag10:=as.numeric(cumulative_dose_lung_lag10)]

df_comb <- df_comb[, exit_adjust_AML:=as.numeric(exit_adjust_AML)]
df_comb <- df_comb[, exit_adjust_AllCancer_excElse:=as.numeric(exit_adjust_AllCancer_excElse)]
df_comb <- df_comb[, exit_adjust_AllLymphHema_excLeukAleuk:=as.numeric(exit_adjust_AllLymphHema_excLeukAleuk)]
df_comb <- df_comb[, exit_adjust_CLL:=as.numeric(exit_adjust_CLL)]
df_comb <- df_comb[, exit_adjust_Esophagus:=as.numeric(exit_adjust_Esophagus)]
df_comb <- df_comb[, exit_adjust_Heart_excIsch:=as.numeric(exit_adjust_Heart_excIsch)]
df_comb <- df_comb[, exit_adjust_Ischemic:=as.numeric(exit_adjust_Ischemic)]
df_comb <- df_comb[, exit_adjust_NonCLL_excAML:=as.numeric(exit_adjust_NonCLL_excAML)]
df_comb <- df_comb[, exit_adjust_Parkinsons:=as.numeric(exit_adjust_Parkinsons)]
df_comb <- df_comb[, exit_adjust_censor:=as.numeric(exit_adjust_censor)]
df_comb <- df_comb[, exit_adjust_lung:=as.numeric(exit_adjust_lung)]

df_comb$exit_adjust_AML                       <- compx_round(df_comb$exit_adjust_AML, digits=3)
df_comb$exit_adjust_AllCancer_excElse         <- compx_round(df_comb$exit_adjust_AllCancer_excElse, digits=3)
df_comb$exit_adjust_AllLymphHema_excLeukAleuk <- compx_round(df_comb$exit_adjust_AllLymphHema_excLeukAleuk, digits=3)
df_comb$exit_adjust_CLL                       <- compx_round(df_comb$exit_adjust_CLL, digits=3)
df_comb$exit_adjust_Esophagus                 <- compx_round(df_comb$exit_adjust_Esophagus, digits=3)
df_comb$exit_adjust_Heart_excIsch             <- compx_round(df_comb$exit_adjust_Heart_excIsch, digits=3)
df_comb$exit_adjust_Ischemic                  <- compx_round(df_comb$exit_adjust_Ischemic, digits=3)
df_comb$exit_adjust_NonCLL_excAML             <- compx_round(df_comb$exit_adjust_NonCLL_excAML, digits=3)
df_comb$exit_adjust_Parkinsons                <- compx_round(df_comb$exit_adjust_Parkinsons, digits=3)
df_comb$exit_adjust_censor                    <- compx_round(df_comb$exit_adjust_censor, digits=3)
df_comb$exit_adjust_lung                      <- compx_round(df_comb$exit_adjust_lung, digits=3)



#
df_comb <- df_comb[, event_t1 := fifelse(event_Parkinsons == 1, exit_adjust_Parkinsons, 1000)]

df_comb <- df_comb[, event_t20 := fifelse(event_Ischemic == 1, exit_adjust_Ischemic, 1000)]
df_comb <- df_comb[, event_t21 := fifelse(event_Heart_excIsch == 1, exit_adjust_Heart_excIsch, 1000)]
df_comb <- df_comb[, event_t2 := ifelse(event_t20 < event_t21, event_t20, event_t21)]



df_comb <- df_comb[, event_t  := fifelse(event_t1 < event_t2, event_t1, event_t2)]

df_comb <- df_comb[, event_t  := fifelse(event_t < 1000, event_t, 1000)]

StudyID_first_event <- df_comb %>% group_by(StudyID) %>% summarise(event_age = min(event_t))
df_comb <- df_comb[StudyID_first_event, on = c("StudyID"), nomatch = 0]
df_comb <- df_comb[age_entry < event_age,]

df_comb <- df_comb[, adjust_heart := ifelse(exit_adjust_Ischemic < exit_adjust_Heart_excIsch, exit_adjust_Ischemic, exit_adjust_Heart_excIsch)]
df_comb <- df_comb[, exit_adjust := fifelse(adjust_heart < exit_adjust_Parkinsons, adjust_heart, exit_adjust_Parkinsons)]
df_comb$event_1 <- as.integer(df_comb$event_Parkinsons == 1)

df_comb$event_2 <- df_comb$event_Ischemic + df_comb$event_Heart_excIsch
df_comb <- df_comb[, event_2 := ifelse(event_2 > 1, 1, 0)]
df_comb$event_2 <- as.integer(df_comb$event_2)

df_comb <- df_comb[, event_1 := fifelse(adjust_heart < exit_adjust_Parkinsons, 0, event_1)]
df_comb <- df_comb[, event_2 := fifelse(exit_adjust_Parkinsons < adjust_heart, 0, event_2)]

df_comb$exit_adjust<- compx_round(df_comb$exit_adjust, digits=2)

df_comb <- df_comb %>% mutate(event = case_when(
                                          event_1 == 1 ~ 1,
                                          event_2 == 1 ~ 2,
                                          .default = 0   
                                  ))
df_comb$event <- factor(df_comb$event, 0:2, labels=c("censor", "pd", "group"))

tryCatch(
    {
        fgdat <- setDT(finegray(Surv(age_entry, exit_adjust, event) ~ . + strata(SES_HighestEd), data=df_comb, etype="pd", id=StudyID))
    },
    error = function(cond) {
        message("Hit an error")
        message(conditionMessage(cond))
    }
)
print("---------------------------------COLOSSUS START---------------------------------------------")
control <- list("Ncores"=16, maxiters=c(1,20), 'verbose'=2)

model <- Cox(age_entry, exit_adjust, event_1) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm)
#ecox0 <- CoxRun(model, df_comb, control = control, norm = "max")

model <- Cox(age_entry, exit_adjust, event_1) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm) + plinear(cumulative_dose_brain_lag10)
a_n <- c(-0.3, -0.1, -0.6, -0.3, 0.001, -0.01, -0.03, 0.03, -0.4, -0.2, -0.2, -1.2, 2)
ecox1 <- CoxRun(model, df_comb, control = control, a_n = a_n, norm = "max")
tryCatch(
    {
        model <- FineGray(fgstart, fgstop, fgstatus, fgwt) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm) + plinear(cumulative_dose_brain_lag10)
        a_n <- ecox1$beta_0
        efg1 <- CoxRun(model, fgdat, control = control, a_n = a_n, norm = "max")
    },
    error = function(cond) {
        message("Hit an error")
        message(conditionMessage(cond))
    }
)

print("---------------------------------COX REGRESSION RESULT---------------------------------")
#print(ecox0)
print(ecox1)
print("---------------------------------FINEGREY REGRESSION RESULT---------------------------------")
tryCatch(
    {
        print(efg1)
    },
    error = function(cond) {
        message("Hit an error")
        message(conditionMessage(cond))
    }
)
###
'''

temp_script_group_cancer = '''library(data.table)
library(dplyr)
library(Colossus)
library(survival)
library(arrow)

options("width"=200)


compx_round <- function(x,digits){
  a <- floor(x)
  b <- x-a
  b <- b*10^digits
  b <- ceiling(b)
  b <- b*10^(-1*digits)
  a+b
}


DS <- arrow::open_dataset(sources = "../WITH_EVENTS/Subj_{count}_{name}")
## Create a scanner
SO <- Scanner$create(DS)
## Load it as n Arrow Table in memory
AT <- SO$ToTable()
## Convert it to an R data frame
df_comb <- as.data.table(AT)


## Convert everything to the column types they need to be
df_comb <- df_comb[, age_entry:=as.numeric(age_entry)]
df_comb <- df_comb[, age_exit:=as.numeric(age_exit)]
df_comb <- df_comb[, sexm:=as.numeric(sexm)]
df_comb <- df_comb[, yob_cat10:=as.numeric(yob_cat10)]

df_comb <- df_comb[, cumulative_dose_brain_lag10:=as.numeric(cumulative_dose_brain_lag10)]
df_comb <- df_comb[, cumulative_dose_colon_lag10:=as.numeric(cumulative_dose_colon_lag10)]
df_comb <- df_comb[, cumulative_dose_esophagus_lag10:=as.numeric(cumulative_dose_esophagus_lag10)]
df_comb <- df_comb[, cumulative_dose_heart_lag10:=as.numeric(cumulative_dose_heart_lag10)]
df_comb <- df_comb[, cumulative_dose_lung_lag10:=as.numeric(cumulative_dose_lung_lag10)]

df_comb <- df_comb[, exit_adjust_AML:=as.numeric(exit_adjust_AML)]
df_comb <- df_comb[, exit_adjust_AllCancer_excElse:=as.numeric(exit_adjust_AllCancer_excElse)]
df_comb <- df_comb[, exit_adjust_AllLymphHema_excLeukAleuk:=as.numeric(exit_adjust_AllLymphHema_excLeukAleuk)]
df_comb <- df_comb[, exit_adjust_CLL:=as.numeric(exit_adjust_CLL)]
df_comb <- df_comb[, exit_adjust_Esophagus:=as.numeric(exit_adjust_Esophagus)]
df_comb <- df_comb[, exit_adjust_Heart_excIsch:=as.numeric(exit_adjust_Heart_excIsch)]
df_comb <- df_comb[, exit_adjust_Ischemic:=as.numeric(exit_adjust_Ischemic)]
df_comb <- df_comb[, exit_adjust_NonCLL_excAML:=as.numeric(exit_adjust_NonCLL_excAML)]
df_comb <- df_comb[, exit_adjust_Parkinsons:=as.numeric(exit_adjust_Parkinsons)]
df_comb <- df_comb[, exit_adjust_censor:=as.numeric(exit_adjust_censor)]
df_comb <- df_comb[, exit_adjust_lung:=as.numeric(exit_adjust_lung)]

df_comb$exit_adjust_AML                       <- compx_round(df_comb$exit_adjust_AML, digits=3)
df_comb$exit_adjust_AllCancer_excElse         <- compx_round(df_comb$exit_adjust_AllCancer_excElse, digits=3)
df_comb$exit_adjust_AllLymphHema_excLeukAleuk <- compx_round(df_comb$exit_adjust_AllLymphHema_excLeukAleuk, digits=3)
df_comb$exit_adjust_CLL                       <- compx_round(df_comb$exit_adjust_CLL, digits=3)
df_comb$exit_adjust_Esophagus                 <- compx_round(df_comb$exit_adjust_Esophagus, digits=3)
df_comb$exit_adjust_Heart_excIsch             <- compx_round(df_comb$exit_adjust_Heart_excIsch, digits=3)
df_comb$exit_adjust_Ischemic                  <- compx_round(df_comb$exit_adjust_Ischemic, digits=3)
df_comb$exit_adjust_NonCLL_excAML             <- compx_round(df_comb$exit_adjust_NonCLL_excAML, digits=3)
df_comb$exit_adjust_Parkinsons                <- compx_round(df_comb$exit_adjust_Parkinsons, digits=3)
df_comb$exit_adjust_censor                    <- compx_round(df_comb$exit_adjust_censor, digits=3)
df_comb$exit_adjust_lung                      <- compx_round(df_comb$exit_adjust_lung, digits=3)



# If there is a parkinsons event, assign the event time to the column
df_comb <- df_comb[, event_t1 := fifelse(event_Parkinsons == 1, exit_adjust_Parkinsons, 1000)]

# Get the event times for each event in the group
df_comb <- df_comb[, event_t20 := fifelse(event_AML == 1, exit_adjust_AML, 1000)]
df_comb <- df_comb[, event_t21 := fifelse(event_CLL == 1, exit_adjust_CLL, 1000)]
df_comb <- df_comb[, event_t22 := fifelse(event_NonCLL_excAML == 1, exit_adjust_NonCLL_excAML, 1000)]
df_comb <- df_comb[, event_t23 := fifelse(event_AllLymphHema_excLeukAleuk == 1, exit_adjust_AllLymphHema_excLeukAleuk, 1000)]
df_comb <- df_comb[, event_t24 := fifelse(event_Esophagus == 1, exit_adjust_Esophagus, 1000)]
df_comb <- df_comb[, event_t25 := fifelse(event_AllCancer_excElse == 1, exit_adjust_AllCancer_excElse, 1000)]
df_comb <- df_comb[, event_t26 := fifelse(event_lung == 1, exit_adjust_lung, 1000)]

# Find the lowest event time
df_comb <- df_comb[, event_t2 := ifelse(event_t20 < event_t21, event_t20, event_t21)]
df_comb <- df_comb[, event_t2 := ifelse(event_t22 < event_t2, event_t22, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t23 < event_t2, event_t23, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t24 < event_t2, event_t24, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t25 < event_t2, event_t25, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t26 < event_t2, event_t26, event_t2)]

# Find the lowest interval end time
df_comb <- df_comb[, adjust_cancer := ifelse(exit_adjust_AML < exit_adjust_CLL, exit_adjust_AML, exit_adjust_CLL)]
df_comb <- df_comb[, adjust_cancer := ifelse(exit_adjust_NonCLL_excAML < adjust_cancer, exit_adjust_NonCLL_excAML, adjust_cancer)]
df_comb <- df_comb[, adjust_cancer := ifelse(exit_adjust_AllLymphHema_excLeukAleuk < adjust_cancer, exit_adjust_AllLymphHema_excLeukAleuk, adjust_cancer)]
df_comb <- df_comb[, adjust_cancer := ifelse(exit_adjust_Esophagus < adjust_cancer, exit_adjust_Esophagus, adjust_cancer)]
df_comb <- df_comb[, adjust_cancer := ifelse(exit_adjust_AllCancer_excElse < adjust_cancer, exit_adjust_AllCancer_excElse, adjust_cancer)]
df_comb <- df_comb[, adjust_cancer := ifelse(exit_adjust_lung < adjust_cancer, exit_adjust_lung, adjust_cancer)]

# Find the lowest event time in each interval
df_comb <- df_comb[, event_t  := fifelse(event_t1 < event_t2, event_t1, event_t2)]

df_comb <- df_comb[, event_t  := fifelse(event_t < 1000, event_t, 1000)]

# Group and solve for the first event time for each subject
StudyID_first_event <- df_comb %>% group_by(StudyID) %>% summarise(event_age = min(event_t))
df_comb <- df_comb[StudyID_first_event, on = c("StudyID"), nomatch = 0]
# Filter to intervals that start before the final event time
df_comb <- df_comb[age_entry < event_age,]

# assign a single column for the interval end times
df_comb <- df_comb[, exit_adjust := fifelse(adjust_cancer < exit_adjust_Parkinsons, adjust_cancer, exit_adjust_Parkinsons)]
# Create column for when parkinsons events occur
df_comb$event_1 <- as.integer(df_comb$event_Parkinsons == 1)

#Create column for when any of the grouped events occur
df_comb$event_2 <- df_comb$event_AML + df_comb$event_CLL + df_comb$event_NonCLL_excAML + df_comb$event_AllLymphHema_excLeukAleuk + df_comb$event_Esophagus + df_comb$event_AllCancer_excElse + df_comb$event_lung
df_comb <- df_comb[, event_2 := ifelse(event_2 > 1, 1, 0)]
df_comb$event_2 <- as.integer(df_comb$event_2)

# If the grouped event happens before the parkinsons event, then that event cannot be observed
df_comb <- df_comb[, event_1 := fifelse(adjust_cancer < exit_adjust_Parkinsons, 0, event_1)]
# If the parkinsons event happens before the grouped event, then that event cannot be observed
df_comb <- df_comb[, event_2 := fifelse(exit_adjust_Parkinsons < adjust_cancer, 0, event_2)]
# Adjust the interval exit time to reduce the number of risk groups
df_comb$exit_adjust<- compx_round(df_comb$exit_adjust, digits=2)

df_comb <- df_comb %>% mutate(event = case_when(
                                          event_1 == 1 ~ 1,
                                          event_2 == 1 ~ 2,
                                          .default = 0   
                                  ))
df_comb$event <- factor(df_comb$event, 0:2, labels=c("censor", "pd", "group"))

tryCatch(
    {
        fgdat <- setDT(finegray(Surv(age_entry, exit_adjust, event) ~ . + strata(SES_HighestEd), data=df_comb, etype="pd", id=StudyID))
    },
    error = function(cond) {
        message("Hit an error")
        message(conditionMessage(cond))
    }
)
print("---------------------------------COLOSSUS START---------------------------------------------")
control <- list("Ncores"=16, maxiters=c(1,20), 'verbose'=2)

model <- Cox(age_entry, exit_adjust, event_1) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm)
#ecox0 <- CoxRun(model, df_comb, control = control, norm = "max")

model <- Cox(age_entry, exit_adjust, event_1) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm) + plinear(cumulative_dose_brain_lag10)
a_n <- c(-0.3, -0.1, -0.6, -0.3, 0.001, -0.01, -0.03, 0.03, -0.4, -0.2, -0.2, -1.2, 2)
ecox1 <- CoxRun(model, df_comb, control = control, a_n = a_n, norm = "max")
tryCatch(
    {
        model <- FineGray(fgstart, fgstop, fgstatus, fgwt) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm) + plinear(cumulative_dose_brain_lag10)
        a_n <- ecox1$beta_0
        efg1 <- CoxRun(model, fgdat, control = control, a_n = a_n, norm = "max")
    },
    error = function(cond) {
        message("Hit an error")
        message(conditionMessage(cond))
    }
)

print("---------------------------------COX REGRESSION RESULT---------------------------------")
#print(ecox0)
print(ecox1)
print("---------------------------------FINEGREY REGRESSION RESULT---------------------------------")
tryCatch(
    {
        print(efg1)
    },
    error = function(cond) {
        message("Hit an error")
        message(conditionMessage(cond))
    }
)
###
'''

temp_script_group_event = '''library(data.table)
library(dplyr)
library(Colossus)
library(survival)
library(arrow)

options("width"=200)


compx_round <- function(x,digits){
  a <- floor(x)
  b <- x-a
  b <- b*10^digits
  b <- ceiling(b)
  b <- b*10^(-1*digits)
  a+b
}


DS <- arrow::open_dataset(sources = "../WITH_EVENTS/Subj_{count}_{name}")
## Create a scanner
SO <- Scanner$create(DS)
## Load it as n Arrow Table in memory
AT <- SO$ToTable()
## Convert it to an R data frame
df_comb <- as.data.table(AT)


## Convert everything to the column types they need to be
df_comb <- df_comb[, age_entry:=as.numeric(age_entry)]
df_comb <- df_comb[, age_exit:=as.numeric(age_exit)]
df_comb <- df_comb[, sexm:=as.numeric(sexm)]
df_comb <- df_comb[, yob_cat10:=as.numeric(yob_cat10)]

df_comb <- df_comb[, cumulative_dose_brain_lag10:=as.numeric(cumulative_dose_brain_lag10)]
df_comb <- df_comb[, cumulative_dose_colon_lag10:=as.numeric(cumulative_dose_colon_lag10)]
df_comb <- df_comb[, cumulative_dose_esophagus_lag10:=as.numeric(cumulative_dose_esophagus_lag10)]
df_comb <- df_comb[, cumulative_dose_heart_lag10:=as.numeric(cumulative_dose_heart_lag10)]
df_comb <- df_comb[, cumulative_dose_lung_lag10:=as.numeric(cumulative_dose_lung_lag10)]

df_comb <- df_comb[, exit_adjust_AML:=as.numeric(exit_adjust_AML)]
df_comb <- df_comb[, exit_adjust_AllCancer_excElse:=as.numeric(exit_adjust_AllCancer_excElse)]
df_comb <- df_comb[, exit_adjust_AllLymphHema_excLeukAleuk:=as.numeric(exit_adjust_AllLymphHema_excLeukAleuk)]
df_comb <- df_comb[, exit_adjust_CLL:=as.numeric(exit_adjust_CLL)]
df_comb <- df_comb[, exit_adjust_Esophagus:=as.numeric(exit_adjust_Esophagus)]
df_comb <- df_comb[, exit_adjust_Heart_excIsch:=as.numeric(exit_adjust_Heart_excIsch)]
df_comb <- df_comb[, exit_adjust_Ischemic:=as.numeric(exit_adjust_Ischemic)]
df_comb <- df_comb[, exit_adjust_NonCLL_excAML:=as.numeric(exit_adjust_NonCLL_excAML)]
df_comb <- df_comb[, exit_adjust_Parkinsons:=as.numeric(exit_adjust_Parkinsons)]
df_comb <- df_comb[, exit_adjust_censor:=as.numeric(exit_adjust_censor)]
df_comb <- df_comb[, exit_adjust_lung:=as.numeric(exit_adjust_lung)]

df_comb$exit_adjust_AML                       <- compx_round(df_comb$exit_adjust_AML, digits=3)
df_comb$exit_adjust_AllCancer_excElse         <- compx_round(df_comb$exit_adjust_AllCancer_excElse, digits=3)
df_comb$exit_adjust_AllLymphHema_excLeukAleuk <- compx_round(df_comb$exit_adjust_AllLymphHema_excLeukAleuk, digits=3)
df_comb$exit_adjust_CLL                       <- compx_round(df_comb$exit_adjust_CLL, digits=3)
df_comb$exit_adjust_Esophagus                 <- compx_round(df_comb$exit_adjust_Esophagus, digits=3)
df_comb$exit_adjust_Heart_excIsch             <- compx_round(df_comb$exit_adjust_Heart_excIsch, digits=3)
df_comb$exit_adjust_Ischemic                  <- compx_round(df_comb$exit_adjust_Ischemic, digits=3)
df_comb$exit_adjust_NonCLL_excAML             <- compx_round(df_comb$exit_adjust_NonCLL_excAML, digits=3)
df_comb$exit_adjust_Parkinsons                <- compx_round(df_comb$exit_adjust_Parkinsons, digits=3)
df_comb$exit_adjust_censor                    <- compx_round(df_comb$exit_adjust_censor, digits=3)
df_comb$exit_adjust_lung                      <- compx_round(df_comb$exit_adjust_lung, digits=3)



# If there is a parkinsons event, assign the event time to the column
df_comb <- df_comb[, event_t1 := fifelse(event_Parkinsons == 1, exit_adjust_Parkinsons, 1000)]

# Get the event times for each event in the group
df_comb <- df_comb[, event_t20 := fifelse(event_AML == 1, exit_adjust_AML, 1000)]
df_comb <- df_comb[, event_t21 := fifelse(event_CLL == 1, exit_adjust_CLL, 1000)]
df_comb <- df_comb[, event_t22 := fifelse(event_NonCLL_excAML == 1, exit_adjust_NonCLL_excAML, 1000)]
df_comb <- df_comb[, event_t23 := fifelse(event_AllLymphHema_excLeukAleuk == 1, exit_adjust_AllLymphHema_excLeukAleuk, 1000)]
df_comb <- df_comb[, event_t24 := fifelse(event_Esophagus == 1, exit_adjust_Esophagus, 1000)]
df_comb <- df_comb[, event_t25 := fifelse(event_AllCancer_excElse == 1, exit_adjust_AllCancer_excElse, 1000)]
df_comb <- df_comb[, event_t26 := fifelse(event_lung == 1, exit_adjust_lung, 1000)]
df_comb <- df_comb[, event_t27 := fifelse(event_Ischemic == 1, exit_adjust_Ischemic, 1000)]
df_comb <- df_comb[, event_t28 := fifelse(event_Heart_excIsch == 1, exit_adjust_Heart_excIsch, 1000)]

# Find the lowest event time
df_comb <- df_comb[, event_t2 := ifelse(event_t20 < event_t21, event_t20, event_t21)]
df_comb <- df_comb[, event_t2 := ifelse(event_t22 < event_t2, event_t22, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t23 < event_t2, event_t23, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t24 < event_t2, event_t24, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t25 < event_t2, event_t25, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t26 < event_t2, event_t26, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t27 < event_t2, event_t27, event_t2)]
df_comb <- df_comb[, event_t2 := ifelse(event_t28 < event_t2, event_t28, event_t2)]

# Find the lowest interval end time
df_comb <- df_comb[, adjust_events := ifelse(exit_adjust_AML < exit_adjust_CLL, exit_adjust_AML, exit_adjust_CLL)]
df_comb <- df_comb[, adjust_events := ifelse(exit_adjust_NonCLL_excAML < adjust_events, exit_adjust_NonCLL_excAML, adjust_events)]
df_comb <- df_comb[, adjust_events := ifelse(exit_adjust_AllLymphHema_excLeukAleuk < adjust_events, exit_adjust_AllLymphHema_excLeukAleuk, adjust_events)]
df_comb <- df_comb[, adjust_events := ifelse(exit_adjust_Esophagus < adjust_events, exit_adjust_Esophagus, adjust_events)]
df_comb <- df_comb[, adjust_events := ifelse(exit_adjust_AllCancer_excElse < adjust_events, exit_adjust_AllCancer_excElse, adjust_events)]
df_comb <- df_comb[, adjust_events := ifelse(exit_adjust_lung < adjust_events, exit_adjust_lung, adjust_events)]
df_comb <- df_comb[, adjust_events := ifelse(exit_adjust_Ischemic < adjust_events, exit_adjust_Ischemic, adjust_events)]
df_comb <- df_comb[, adjust_events := ifelse(exit_adjust_Heart_excIsch < adjust_events, exit_adjust_Heart_excIsch, adjust_events)]

# Find the lowest event time in each interval
df_comb <- df_comb[, event_t  := fifelse(event_t1 < event_t2, event_t1, event_t2)]

df_comb <- df_comb[, event_t  := fifelse(event_t < 1000, event_t, 1000)]

# Group and solve for the first event time for each subject
StudyID_first_event <- df_comb %>% group_by(StudyID) %>% summarise(event_age = min(event_t))
df_comb <- df_comb[StudyID_first_event, on = c("StudyID"), nomatch = 0]
# Filter to intervals that start before the final event time
df_comb <- df_comb[age_entry < event_age,]

# assign a single column for the interval end times
df_comb <- df_comb[, exit_adjust := fifelse(adjust_events < exit_adjust_Parkinsons, adjust_events, exit_adjust_Parkinsons)]
# Create column for when parkinsons events occur
df_comb$event_1 <- as.integer(df_comb$event_Parkinsons == 1)

#Create column for when any of the grouped events occur
df_comb$event_2 <- df_comb$event_AML + df_comb$event_CLL + df_comb$event_NonCLL_excAML + df_comb$event_AllLymphHema_excLeukAleuk + df_comb$event_Esophagus + df_comb$event_AllCancer_excElse + df_comb$event_lung + df_comb$event_Ischemic + df_comb$event_Heart_excIsch
df_comb <- df_comb[, event_2 := ifelse(event_2 > 1, 1, 0)]
df_comb$event_2 <- as.integer(df_comb$event_2)

# If the grouped event happens before the parkinsons event, then that event cannot be observed
df_comb <- df_comb[, event_1 := fifelse(adjust_events < exit_adjust_Parkinsons, 0, event_1)]
# If the parkinsons event happens before the grouped event, then that event cannot be observed
df_comb <- df_comb[, event_2 := fifelse(exit_adjust_Parkinsons < adjust_events, 0, event_2)]
# Adjust the interval exit time to reduce the number of risk groups
df_comb$exit_adjust<- compx_round(df_comb$exit_adjust, digits=2)

df_comb <- df_comb %>% mutate(event = case_when(
                                          event_1 == 1 ~ 1,
                                          event_2 == 1 ~ 2,
                                          .default = 0   
                                  ))
df_comb$event <- factor(df_comb$event, 0:2, labels=c("censor", "pd", "group"))

tryCatch(
    {
        fgdat <- setDT(finegray(Surv(age_entry, exit_adjust, event) ~ . + strata(SES_HighestEd), data=df_comb, etype="pd", id=StudyID))
    },
    error = function(cond) {
        message("Hit an error")
        message(conditionMessage(cond))
    }
)
print("---------------------------------COLOSSUS START---------------------------------------------")
control <- list("Ncores"=16, maxiters=c(1,20), 'verbose'=2)

model <- Cox(age_entry, exit_adjust, event_1) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm)
#ecox0 <- CoxRun(model, df_comb, control = control, norm = "max")

model <- Cox(age_entry, exit_adjust, event_1) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm) + plinear(cumulative_dose_brain_lag10)
a_n <- c(-0.3, -0.1, -0.6, -0.3, 0.001, -0.01, -0.03, 0.03, -0.4, -0.2, -0.2, -1.2, 2)
ecox1 <- CoxRun(model, df_comb, control = control, a_n = a_n, norm = "max")
tryCatch(
    {
        model <- FineGray(fgstart, fgstop, fgstatus, fgwt) ~ loglinear(factor(SES_HighestEd), factor(yob_cat10), sexm) + plinear(cumulative_dose_brain_lag10)
        a_n <- ecox1$beta_0
        efg1 <- CoxRun(model, fgdat, control = control, a_n = a_n, norm = "max")
    },
    error = function(cond) {
        message("Hit an error")
        message(conditionMessage(cond))
    }
)

print("---------------------------------COX REGRESSION RESULT---------------------------------")
#print(ecox0)
print(ecox1)
print("---------------------------------FINEGREY REGRESSION RESULT---------------------------------")
tryCatch(
    {
        print(efg1)
    },
    error = function(cond) {
        message("Hit an error")
        message(conditionMessage(cond))
    }
)
###
'''

bash_script = "cd ANALYSIS_SCRIPTS\n"

doses = {'Esophagus':"cumulative_dose_esophagus_lag10",
         'lung':"cumulative_dose_lung_lag10",
         'Parkinsons':"cumulative_dose_brain_lag10",
         'Ischemic':"cumulative_dose_heart_lag10",
         'CLL':"cumulative_dose_bonem_lag10",
         'AML':"cumulative_dose_colon_lag10",
         'NonCLL_excAML':"cumulative_dose_colon_lag10",
         "Heart_excIsch":"cumulative_dose_heart_lag10",
         "AllLymphHema_excLeukAleuk":"cumulative_dose_colon_lag10",
         "AllCancer_excElse":"cumulative_dose_colon_lag10"
}

outcome_single = list(doses.keys())
doses_single = list(doses.values())
outcome_short = ["eso", "lung", "park", "isc", "cll", "aml", "ncll", "HeIs", "ALeLA", "ACeE"]

#print(outcome_single)
#print(doses_single)
#
#assert False
bash_script += 'echo "All and Events"\n'
for i in range(500):
    for j_name in ["CONST","PARA","RF"]:
        bash_script += "echo " + str(i) + " " + j_name + "\n"
        ## All events
        fname = "ANALYSIS_SCRIPTS/Analyze_"+str(i)+"_"+j_name+"_event_group.R"
        with open(fname,'w') as f:
            f.write(temp_script_group_event.replace("{count}",str(i)).replace("{name}",str(j_name)))
        bash_script += "Rscript " + "Analyze_"+str(i)+"_"+j_name+"_event_group.R" + " &> ../RESULTS/COX_" + str(i)+"_"+j_name+"_event_group.txt\n"
bash_script += 'echo "Groups"\n'
for i in range(500):
    for j_name in ["CONST","PARA","RF"]:
        bash_script += "echo " + str(i)+"_"+j_name + "\n"
        ## All heart disease
        fname = "ANALYSIS_SCRIPTS/Analyze_"+str(i)+"_"+j_name+"_heart_group.R"
        with open(fname,'w') as f:
            f.write(temp_script_group_heart.replace("{count}",str(i)).replace("{name}",str(j_name)))
        bash_script += "Rscript " + "Analyze_"+str(i)+"_"+j_name+"_heart_group.R" + " &> ../RESULTS/COX_" + str(i)+"_"+j_name+"_heart_group.txt\n"
        ## All cancer
        fname = "ANALYSIS_SCRIPTS/Analyze_"+str(i)+"_"+j_name+"_cancer_group.R"
        with open(fname,'w') as f:
            f.write(temp_script_group_cancer.replace("{count}",str(i)).replace("{name}",str(j_name)))
        bash_script += "Rscript " + "Analyze_"+str(i)+"_"+j_name+"_cancer_group.R" + " &> ../RESULTS/COX_" + str(i)+"_"+j_name+"_cancer_group.txt\n"
bash_script += 'echo "Pairs and Single"\n'
for i in range(500):
    for j_name in ["CONST","PARA","RF"]:
        bash_script += "echo " + str(i)+"_"+j_name + "\n"
        # Single and Pair Regressions
        for j in [2]:
            # Doing the single with parkinsons
            fname = "ANALYSIS_SCRIPTS/Analyze_"+str(i)+"_"+j_name+"_" + outcome_short[j] + "_single.R"
            with open(fname,'w') as f:
                f.write(temp_script_single.replace("{count}",str(i)).replace("{name}",str(j_name)).replace("{outcome}",outcome_single[j]).replace("{dose}",doses_single[j]))
            bash_script += "Rscript " + "Analyze_"+str(i)+"_"+j_name+"_" + outcome_short[j] + "_single.R" + " &> ../RESULTS/COX_" + str(i)+"_"+j_name+"_" + outcome_short[j] + "_single.txt\n"

with open("run_analysis.sh",'w') as f:
    f.write(bash_script)

