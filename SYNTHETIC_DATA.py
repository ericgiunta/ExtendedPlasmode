import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1022026)

# This file produces source files with similar types of data as the original files
# NOTE THE SYNTHETIC DATA WILL NOT REPRODUCE PAPER RESULTS, AND WILL NOT MATCH THE HARD CODED PARAMETRIC DISTRIBUTIONS

# NPP23A01_1.csv
#'StudyID', 'SES_HighestEd', 'YOH', "YOB", 'duremp', 'sexm', 'SeqNo', "AllCancer", "Esophagus", "lung", "AllLymphaticHematopoietic", "Parkinsons", "HeartDis", "Ischemic", "NonCLL", "LeukemiaAleukemia", "CLL", "AML"
# NPP23A01_4.csv
#'StudyID', 'age_entry', 'age_exit', 'cumulative_dose_brain_lag10', 'SeqNo'
# NPP23A01_5.csv
#'StudyID', 'age_entry', 'age_exit', 'cumulative_dose_colon_lag10', 'SeqNo', 'year'
# NPP23A01_6.csv
#'StudyID', 'year', 'age_entry', 'age_exit', 'cumulative_dose_esophagus_lag10', 'SeqNo'
# NPP23A01_7.csv
#'StudyID', 'year', 'age_entry', 'age_exit', 'cumulative_dose_lung_lag10', 'SeqNo'
# NPP23A01_8.csv
#'StudyID', 'age_entry', 'age_exit', 'cumulative_dose_bonem_lag10', 'SeqNo'
# NPP23A01_9.csv
#'StudyID', 'age_entry', 'age_exit', 'cumulative_dose_heart_lag10', 'SeqNo'

# Person Info
studyid_person = []
ses_highest = []
yoh = []
yob = []
duremp = []
sexm = []
AllCancer = []
Esophagus = []
lung = []
AllLymphaticHematopoietic = []
Parkinsons = []
HeartDis = []
Ischemic = []
NonCLL = []
LeukemiaAleukemia = []
CLL = []
AML = []

# Annual data
StudyID = []
age_entry = []
age_exit = []
year =  []
cumulative_dose_brain_lag10 = []
cumulative_dose_colon_lag10 = []
cumulative_dose_esophagus_lag10 = []
cumulative_dose_lung_lag10 = []
cumulative_dose_bonem_lag10 = []
cumulative_dose_heart_lag10 = []

# Now we randomly generate data
N = 100 # number of subjects
# Randomly sample the constant data
studyid_person = ["SUBJ_"+"0"*(len(str(N+1)) - len(str(i+1))) + str(i+1) for i in range(N)] # Study ID
ses_highest = np.random.choice(["Low", "High", "Medium"], N, True, [0.24, 0.26, 0.50]) # sampling with total probability
yob = np.random.randint(1916, 1956, N) # uniform across the year of birth bins
yoh = np.random.randint(18,30, N) + yob # subjects are hired between age 18 and age 30
duremp = np.random.randint(10,40, N) # subjects work between 10 and 40 years
sexm = np.random.choice([1, 0], N, True, [0.03, 0.97]) # sampling with total probability

event_status = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], N, True, [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.78]) # randomly assigning one of the events, or censoring
# Some events are overlapping, i.e. all cancers includes lung, esophagus, and all Lymphatic and Hematopoietic
Parkinsons = np.where(event_status == 1, 1, 0)
Ischemic = np.where(event_status == 2, 1, 0)
HeartDis = np.select([event_status == 2, event_status == 3], [1, 1], default = 0) # Heart disease is also true when ischemic is true
AML = np.where(event_status == 4, 1, 0)
CLL = np.where(event_status == 5, 1, 0)
NonCLL = np.select([event_status == 5, event_status == 6], [1, 1], default = 0)
LeukemiaAleukemia = np.select([NonCLL == 1, event_status == 7], [1, 1], default = 0)
AllLymphaticHematopoietic = np.select([LeukemiaAleukemia == 1, event_status == 8], [1, 1], default = 0)
Esophagus = np.where(event_status == 9, 1, 0)
lung = np.where(event_status == 10, 1, 0)
AllCancer = np.select([AllLymphaticHematopoietic == 1, Esophagus == 1, lung == 1, event_status == 11], [1, 1, 1, 1], default = 0)

# Now we have all of the constant data, we can generate the annual data
for i in range(N):
    SID = studyid_person[i]
    age0 = yoh[i] - yob[i]
    year0 = yoh[i]
    dur_tot = duremp[i]
    # Suppose we stop at a random age before 95 or the year 2000
    dur_age = np.random.randint(dur_tot+10, 95-age0)
    dur_2000 = 2000 - year0
    dur_used = np.min([dur_age, dur_2000])
    # We need to accumulate the doses
    annual_brain = 0
    annual_colon = 0
    annual_esophagus = 0
    annual_lung = 0
    annual_bonem = 0
    annual_heart = 0
    for j in range(dur_used):
        StudyID.append(SID)
        age_entry.append(age0+j)
        age_exit.append(age0+j+1)
        year.append(year0+j)
        # now we need to randomly assign doses
        # Each dose can be zero or non-dose, then ranging from 1e-3 to 1e3
        heart_dose = np.random.choice([0,1], 1, True, [0.25, 0.75]) * np.random.rand(1) * np.random.choice([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 1, True, [0.01, 0.1, 0.2, 0.38, 0.2, 0.1, 0.01])
        nonheart_dose = np.random.choice([0,1], 1, True, [0.25, 0.75]) * np.random.rand(1) * np.random.choice([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 1, True, [0.01, 0.1, 0.2, 0.38, 0.2, 0.1, 0.01])
        # Now we accumulate heart
        annual_heart += heart_dose
        # We slightly vary the nonheart doses
        #               general dose           1% chance to be zero regardless                # varying by 5%
        annual_brain += nonheart_dose * np.random.choice([0,1], 1, True, [0.01, 0.99]) * (1 + 0.1*(np.random.rand()-0.5))
        annual_colon += nonheart_dose * np.random.choice([0,1], 1, True, [0.01, 0.99]) * (1 + 0.1*(np.random.rand()-0.5))
        annual_esophagus += nonheart_dose * np.random.choice([0,1], 1, True, [0.01, 0.99]) * (1 + 0.1*(np.random.rand()-0.5))
        annual_lung += nonheart_dose * np.random.choice([0,1], 1, True, [0.01, 0.99]) * (1 + 0.1*(np.random.rand()-0.5))
        annual_bonem += nonheart_dose * np.random.choice([0,1], 1, True, [0.01, 0.99]) * (1 + 0.1*(np.random.rand()-0.5))
        #
        cumulative_dose_brain_lag10.append(annual_brain)
        cumulative_dose_colon_lag10.append(annual_colon)
        cumulative_dose_esophagus_lag10.append(annual_esophagus)
        cumulative_dose_lung_lag10.append(annual_lung)
        cumulative_dose_bonem_lag10.append(annual_bonem)
        cumulative_dose_heart_lag10.append(annual_heart)

df_person = pd.DataFrame({'StudyID':studyid_person,
    'SES_HighestEd':ses_highest,
    'YOH':yoh,
    "YOB":yob,
    'duremp':duremp,
    'sexm':sexm,
    'SeqNo':range(N),
    "AllCancer":AllCancer,
    "Esophagus":Esophagus,
    "lung":lung,
    "AllLymphaticHematopoietic":AllLymphaticHematopoietic,
    "Parkinsons":Parkinsons,
    "HeartDis":HeartDis,
    "Ischemic":Ischemic,
    "NonCLL":NonCLL,
    "LeukemiaAleukemia":LeukemiaAleukemia,
    "CLL":CLL,
    "AML":AML
})

df_brain = pd.DataFrame({
    'StudyID':StudyID,
    'age_entry':age_entry,
    'age_exit':age_exit,
    'cumulative_dose_brain_lag10':cumulative_dose_brain_lag10,
    'SeqNo':range(len(age_entry))
})
df_colon = pd.DataFrame({
    'StudyID':StudyID,
    'age_entry':age_entry,
    'age_exit':age_exit,
    'cumulative_dose_colon_lag10':cumulative_dose_colon_lag10,
    'SeqNo':range(len(age_entry))
})
df_eso = pd.DataFrame({
    'StudyID':StudyID,
    'age_entry':age_entry,
    'age_exit':age_exit,
    'cumulative_dose_esophagus_lag10':cumulative_dose_esophagus_lag10,
    'SeqNo':range(len(age_entry))
})
df_lung = pd.DataFrame({
    'StudyID':StudyID,
    'age_entry':age_entry,
    'age_exit':age_exit,
    'cumulative_dose_lung_lag10':cumulative_dose_lung_lag10,
    'SeqNo':range(len(age_entry))
})
df_bonem = pd.DataFrame({
    'StudyID':StudyID,
    'age_entry':age_entry,
    'age_exit':age_exit,
    'cumulative_dose_bonem_lag10':cumulative_dose_bonem_lag10,
    'SeqNo':range(len(age_entry))
})
df_heart = pd.DataFrame({
    'StudyID':StudyID,
    'age_entry':age_entry,
    'age_exit':age_exit,
    'cumulative_dose_heart_lag10':cumulative_dose_heart_lag10,
    'SeqNo':range(len(age_entry))
})

df_person.to_csv("SIM_NPP23A01_1.csv")
df_brain.to_csv("SIM_NPP23A01_4.csv")
df_colon.to_csv("SIM_NPP23A01_5.csv")
df_eso.to_csv("SIM_NPP23A01_6.csv")
df_lung.to_csv("SIM_NPP23A01_7.csv")
df_bonem.to_csv("SIM_NPP23A01_8.csv")
df_heart.to_csv("SIM_NPP23A01_9.csv")
