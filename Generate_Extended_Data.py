import polars as pl
import numpy as np
import os
import glob

import pandas as pd
from scipy.stats import triang
from scipy.stats import multivariate_normal

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# We start by preping the data, putting into one big table
df_colon  = pl.read_csv("SIM_NPP23A01_5.csv")
df_heart  = pl.read_csv("SIM_NPP23A01_9.csv")
df_brain  = pl.read_csv("SIM_NPP23A01_4.csv")
df_person = pl.read_csv("SIM_NPP23A01_1.csv")
df_bone   = pl.read_csv("SIM_NPP23A01_8.csv")
df_lung   = pl.read_csv("SIM_NPP23A01_7.csv")
df_esop   = pl.read_csv("SIM_NPP23A01_6.csv")

## Filter the tables into the necessary columns
#
colon_columns = ['StudyID', 'age_entry', 'age_exit', 'cumulative_dose_colon_lag10', 'SeqNo', 'year']
df_colon = df_colon.select(colon_columns)
#
heart_columns = ['StudyID', 'age_entry', 'age_exit', 'cumulative_dose_heart_lag10', 'SeqNo']
df_heart = df_heart.select(heart_columns)
#
brain_columns = ['StudyID', 'age_entry', 'age_exit', 'cumulative_dose_brain_lag10', 'SeqNo']
df_brain = df_brain.select(brain_columns)
#
bone_columns = ['StudyID', 'age_entry', 'age_exit', 'cumulative_dose_bonem_lag10', 'SeqNo']
df_bone = df_bone.select(bone_columns)
#
lung_columns = ['StudyID', 'year', 'age_entry', 'age_exit', 'cumulative_dose_lung_lag10', 'SeqNo']
df_lung = df_lung.select(lung_columns)
#
esop_columns = ['StudyID', 'year', 'age_entry', 'age_exit', 'cumulative_dose_esophagus_lag10', 'SeqNo']
df_esop = df_esop.select(esop_columns)
#
df_person = df_person.with_columns(
    yob_cat10 = pl.col("YOB").cut(breaks = [1916, 1921, 1926, 1931, 1936, 1941, 1946, 1951, 1956], labels = [str(i) for i in [0,1,2,3,4,5,6,7,8,9]]),
    yob_cat6 = pl.col("YOB").cut(breaks = [1920, 1930, 1940, 1950, 1960], labels = [str(i) for i in [0,1,2,3,4,5]])
    )
df_person = df_person.with_columns(
    NonCLL_excAML = pl.col("NonCLL") - pl.col("AML"),
    Heart_excIsch = pl.col("HeartDis") - pl.col("Ischemic"),
    AllLymphHema_excLeukAleuk = pl.col("AllLymphaticHematopoietic") - pl.col("LeukemiaAleukemia"),
    AllCancer_excElse = pl.col("AllCancer") - pl.col("Esophagus") - pl.col("lung")  - pl.col("AllLymphaticHematopoietic")
)
df_person = df_person.with_columns(
    NonCLL_excAML = pl.when(pl.col("NonCLL_excAML") < 1).then(pl.lit(0)).otherwise(pl.lit(1)),
    Heart_excIsch = pl.when(pl.col("Heart_excIsch") < 1).then(pl.lit(0)).otherwise(pl.lit(1)),
    AllLymphHema_excLeukAleuk = pl.when(pl.col("AllLymphHema_excLeukAleuk") < 1).then(pl.lit(0)).otherwise(pl.lit(1)),
    AllCancer_excElse = pl.when(pl.col("AllCancer_excElse") < 1).then(pl.lit(0)).otherwise(pl.lit(1))
)

df_person = df_person.with_columns(
    allevent = pl.col('Esophagus') + pl.col('lung') + pl.col('Parkinsons') + pl.col('Ischemic') + pl.col('CLL') + pl.col('AML') + pl.col('NonCLL_excAML') + pl.col("Heart_excIsch") + pl.col("AllLymphHema_excLeukAleuk") + pl.col("AllCancer_excElse")
)
df_person = df_person.with_columns(
    allevent = pl.when(pl.col("allevent") > 0).then(pl.lit(1)).otherwise(pl.lit(0))
)
df_person = df_person.with_columns(
    censored = pl.lit(1) - pl.col("allevent")
)
person_columns = ['StudyID', 'yob_cat10', 'yob_cat6', 'SES_HighestEd', 'YOH', "YOB", 'duremp', 'sexm', 'SeqNo', 'Esophagus', 'lung', 'Parkinsons', 'Ischemic', 'CLL', 'AML', 'NonCLL_excAML', "Heart_excIsch", "AllLymphHema_excLeukAleuk", "AllCancer_excElse", "allevent", "censored"]

df_person = df_person.select(person_columns)

## Join everything together
#
df_organ = df_colon.join(df_heart.select(['cumulative_dose_heart_lag10',    'SeqNo']), on = 'SeqNo')
df_organ = df_organ.join(df_brain.select(['cumulative_dose_brain_lag10',    'SeqNo']), on = 'SeqNo')
df_organ = df_organ.join(df_bone.select(['cumulative_dose_bonem_lag10',     'SeqNo']), on = 'SeqNo')
df_organ = df_organ.join(df_lung.select(['cumulative_dose_lung_lag10',      'SeqNo']), on = 'SeqNo')
df_organ = df_organ.join(df_esop.select(['cumulative_dose_esophagus_lag10', 'SeqNo']), on = 'SeqNo')
#
df_total = df_organ.join(df_person.select(['StudyID', 'yob_cat10', 'yob_cat6', 'YOB', 'SES_HighestEd', 'sexm', 'YOH', 'duremp', 'Esophagus', 'lung', 'Parkinsons', 'Ischemic', 'CLL', 'AML', 'NonCLL_excAML', "Heart_excIsch", "AllLymphHema_excLeukAleuk", "AllCancer_excElse", "allevent", "censored"]), on = "StudyID")

df_total = df_total.with_columns(
    age_exit = pl.col("age_exit").round(1),
    age_entry = pl.col("age_entry").round(1),
)
total_columns = ['StudyID', 'age_entry', 'age_exit', 'yob_cat10', 'yob_cat6', 'YOH', 'YOB', 'year', 'duremp', 'SES_HighestEd', 'sexm', 'cumulative_dose_colon_lag10', 'cumulative_dose_heart_lag10', 'cumulative_dose_brain_lag10', 'cumulative_dose_bonem_lag10', 'cumulative_dose_lung_lag10', 'cumulative_dose_esophagus_lag10', 'Esophagus', 'lung', 'Parkinsons', 'Ischemic', 'CLL', 'AML', 'NonCLL_excAML', "Heart_excIsch", "AllLymphHema_excLeukAleuk", "AllCancer_excElse", "allevent", "censored", 'SeqNo']
df_total = df_total.select(total_columns)
df_total.write_csv("df_total.csv")

# Further analysis does not need the events
total_columns = ['StudyID', 'age_entry', 'age_exit', 'yob_cat10', 'yob_cat6', 'YOH', 'YOB', 'year', 'duremp', 'SES_HighestEd', 'sexm', 'cumulative_dose_colon_lag10', 'cumulative_dose_heart_lag10', 'cumulative_dose_brain_lag10', 'cumulative_dose_bonem_lag10', 'cumulative_dose_lung_lag10', 'cumulative_dose_esophagus_lag10', 'SeqNo']
df_total = df_total.select(total_columns)
## ------------------------------------------------------------ PULL OUT THE MGY DOSE, NOT THE MSV DOSE ----------------------------------------- ##
df_minage = df_total.select(["StudyID", "age_exit"]).group_by("StudyID").min()
df_minage = df_minage.with_columns(age_min = pl.col("age_exit"))
df_total = df_total.join(df_minage.select(["StudyID", "age_min"]), on = "StudyID")

df_maxage = df_total.select(["StudyID", "age_exit"]).group_by("StudyID").max()
df_maxage = df_maxage.with_columns(age_max = pl.col("age_exit"))
df_total = df_total.join(df_maxage.select(["StudyID", "age_max"]), on = "StudyID")

df_total = df_total.with_columns(brain_prev = np.array([0] + list(df_total["cumulative_dose_brain_lag10"])[:-1]))
df_total = df_total.with_columns(annual_dose_mgy_brain_lag10 = pl.col("cumulative_dose_brain_lag10") - pl.col("brain_prev"))

df_total = df_total.with_columns(lung_prev = np.array([0] + list(df_total["cumulative_dose_lung_lag10"])[:-1]))
df_total = df_total.with_columns(annual_dose_mgy_lung_lag10 = pl.col("cumulative_dose_lung_lag10") - pl.col("lung_prev"))

df_total = df_total.with_columns(colon_prev = np.array([0] + list(df_total["cumulative_dose_colon_lag10"])[:-1]))
df_total = df_total.with_columns(annual_dose_mgy_colon_lag10 = pl.col("cumulative_dose_colon_lag10") - pl.col("colon_prev"))

df_total = df_total.with_columns(heart_prev = np.array([0] + list(df_total["cumulative_dose_heart_lag10"])[:-1]))
df_total = df_total.with_columns(annual_dose_mgy_heart_lag10 = pl.col("cumulative_dose_heart_lag10") - pl.col("heart_prev"))

df_total = df_total.with_columns(bone_prev = np.array([0] + list(df_total["cumulative_dose_bonem_lag10"])[:-1]))
df_total = df_total.with_columns(annual_dose_mgy_bonem_lag10 = pl.col("cumulative_dose_bonem_lag10") - pl.col("bone_prev"))

df_total = df_total.with_columns(esophagus_prev = np.array([0] + list(df_total["cumulative_dose_esophagus_lag10"])[:-1]))
df_total = df_total.with_columns(annual_dose_mgy_esophagus_lag10 = pl.col("cumulative_dose_esophagus_lag10") - pl.col("esophagus_prev"))

for col in ["annual_dose_mgy_brain_lag10","annual_dose_mgy_lung_lag10", "annual_dose_mgy_colon_lag10", "annual_dose_mgy_heart_lag10", "annual_dose_mgy_bonem_lag10", "annual_dose_mgy_esophagus_lag10"]:
    df_total = df_total.with_columns(
        (pl.when(
            pl.col("age_entry") > pl.col("age_min")
        ).then(
            pl.col(col)
        ).otherwise(
            pl.lit(0.0)
        )).alias(col)
    ) # we can't calculate annual dose for the first year
    df_total = df_total.with_columns(
        (pl.when(
            pl.col(col) > 0.0
        ).then(
            pl.col(col)
        ).otherwise(
            pl.lit(0.0)
        ).alias(col))
    ) # A small amount of annual doses are negative? assumed to be an error and set to zero

dbins = [0]+list(np.logspace(-5,np.log10(650),40))
dbins = [float("{:.3e}".format(i)) for i in dbins]

# Set the next years annual dose as a column
df_total = df_total.with_columns(brain_next = np.array(list(df_total["annual_dose_mgy_brain_lag10"])[1:] + [0]))
df_total = df_total.with_columns(lung_next = np.array(list(df_total["annual_dose_mgy_lung_lag10"])[1:] + [0]))
df_total = df_total.with_columns(colon_next = np.array(list(df_total["annual_dose_mgy_colon_lag10"])[1:] + [0]))
df_total = df_total.with_columns(heart_next = np.array(list(df_total["annual_dose_mgy_heart_lag10"])[1:] + [0]))
df_total = df_total.with_columns(bone_next = np.array(list(df_total["annual_dose_mgy_bonem_lag10"])[1:] + [0]))
df_total = df_total.with_columns(esophagus_next = np.array(list(df_total["annual_dose_mgy_esophagus_lag10"])[1:] + [0]))

# Assign dose bins for current year dose
df_total = df_total.with_columns(
    brain_bin = pl.col("annual_dose_mgy_brain_lag10").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    lung_bin  = pl.col("annual_dose_mgy_lung_lag10").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    heart_bin = pl.col("annual_dose_mgy_heart_lag10").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    bone_bin = pl.col("annual_dose_mgy_bonem_lag10").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    colon_bin = pl.col("annual_dose_mgy_colon_lag10").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    esophagus_bin = pl.col("annual_dose_mgy_esophagus_lag10").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8)
)
# Assign dose bins for next year dose
df_total = df_total.with_columns(
    brain_bin_next = pl.col("brain_next").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    lung_bin_next  = pl.col("lung_next").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    heart_bin_next = pl.col("heart_next").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    bone_bin_next = pl.col("bone_next").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    colon_bin_next = pl.col("colon_next").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8),
    esophagus_bin_next = pl.col("esophagus_next").cut(dbins, left_closed = True, labels = [str(i) for i in range(len(dbins)+1)]).cast(pl.String).cast(pl.Int8)
)
# Set the SES status to a number for machine learning purposes
df_total = df_total.with_columns(SES_HighestEd = pl.when(
                                                    pl.col("SES_HighestEd") == "High"
                                                ).then(
                                                    pl.lit(2)
                                                ).when(
                                                    pl.col("SES_HighestEd") == "Medium"
                                                ).then(
                                                    pl.lit(1)
                                                ).otherwise(
                                                    pl.lit(0)
                                                )
                                )
# We can't predict next year annual dose at maximum age either
df_total = df_total.filter(pl.col("age_exit") < pl.col("age_max"))

keep = ['StudyID', 'age_entry', 'age_exit', 'yob_cat10', 'yob_cat6', 'SES_HighestEd', 'sexm', 'YOH', 'YOB', 'year', 'duremp', 'brain_prev', 'annual_dose_mgy_brain_lag10', 'lung_prev', 'annual_dose_mgy_lung_lag10', 'colon_prev', 'annual_dose_mgy_colon_lag10', 'heart_prev', 'annual_dose_mgy_heart_lag10', 'bone_prev', 'annual_dose_mgy_bonem_lag10', 'esophagus_prev', 'annual_dose_mgy_esophagus_lag10', 'brain_next', 'lung_next', 'colon_next', 'heart_next', 'bone_next', 'esophagus_next', 'brain_bin', 'lung_bin', 'heart_bin', 'bone_bin', 'colon_bin', 'esophagus_bin', 'brain_bin_next', 'lung_bin_next', 'heart_bin_next', 'bone_bin_next', 'colon_bin_next', 'esophagus_bin_next']
df_total = df_total.select(keep)
# Data saved, so that the previous steps don't need to be repeated
df_total.write_csv("preped_data.csv")

## --------------------------------------------- Next sections cover the steps for generating the newly observed data, up to age 95 ----------------------------- ##

## ---------------------------------------------- First the original data --------------------------------------------------------------------------------------- ##
df_tot = pd.read_csv("preped_data.csv")
col_keep = ['StudyID', 'age_entry',
       'age_exit', 'SES_HighestEd',
       'sexm', 'yob_cat10', 'yob_cat6',
       'annual_dose_mgy_colon_lag10', 'cumulative_dose_colon_lag10',
       'annual_dose_mgy_heart_lag10', 'cumulative_dose_heart_lag10',
       'annual_dose_mgy_brain_lag10', 'cumulative_dose_brain_lag10',
       'annual_dose_mgy_bonem_lag10', 'cumulative_dose_bonem_lag10',
       'annual_dose_mgy_lung_lag10', 'cumulative_dose_lung_lag10',
       'annual_dose_mgy_esophagus_lag10', 'cumulative_dose_esophagus_lag10',
       ]
df_tot.sort_values(['StudyID','age_entry'], ascending=[True,True],inplace=True)
#
for col in ['colon', 'heart', 'brain', 'bonem', 'lung', 'esophagus']:
    name0 = 'cumulative_dose_' + col + '_lag10'
    name1 = 'annual_dose_mgy_' + col + '_lag10'
    df_tot[name0] = df_tot.groupby('StudyID')[name1].cumsum() / 1000
#
df_tot[col_keep].to_csv("OG_data.csv")

## --------------------------------------------- Extend with constant ----------------------------------------------------- ##

df_tot = pd.read_csv("preped_data.csv")
dbins = [0]+list(np.logspace(-5,np.log10(650),40))
dbins = [float("{:.3e}".format(i)) for i in dbins]

df_dose_maxage = df_tot.groupby("StudyID").max()
df_dose_maxage = df_dose_maxage[["age_entry"]]
df_dose_maxage.columns = ["age_max"]
df_person = df_tot.join(df_dose_maxage, on = "StudyID")

subj_ids = df_person.StudyID
rng = np.random.default_rng(seed = 374221226)

col_keep = ['StudyID', 'age_entry',
       'age_exit', 'SES_HighestEd',
       'sexm', 'yob_cat10', 'yob_cat6',
       'annual_dose_mgy_colon_lag10', 'cumulative_dose_colon_lag10',
       'annual_dose_mgy_heart_lag10', 'cumulative_dose_heart_lag10',
       'annual_dose_mgy_brain_lag10', 'cumulative_dose_brain_lag10',
       'annual_dose_mgy_bonem_lag10', 'cumulative_dose_bonem_lag10',
       'annual_dose_mgy_lung_lag10', 'cumulative_dose_lung_lag10',
       'annual_dose_mgy_esophagus_lag10', 'cumulative_dose_esophagus_lag10',
       ]
N = 1

for subj_i in range(N):
    print("Dataset: " + str(subj_i))
    #
    df_person_start = df_person.copy()
    # --------------------------------------------------------------------------------------------------------------------------- #

    df_gen    = df_person_start[df_person_start["age_entry"] <= df_person_start["age_max"]].copy()
    df_next = df_person_start[df_person_start["age_entry"] == df_person_start["age_max"]].copy()

    df_next["colon_next"] = 0.0
    df_next["heart_next"] = 0.0
    df_next["brain_next"] = 0.0
    df_next["bone_next"] = 0.0
    df_next["lung_next"] = 0.0
    df_next["esophagus_next"] = 0.0
    #
    df_next["colon_bin_next"] = 0.0
    df_next["heart_bin_next"] = 0.0
    df_next["brain_bin_next"] = 0.0
    df_next["bone_bin_next"] = 0.0
    df_next["lung_bin_next"] = 0.0
    df_next["esophagus_bin_next"] = 0.0
    #
    df_next.drop(['age_max'], axis = 1, inplace = True)
    df_gen.drop(['age_max'], axis = 1, inplace = True)
    count = 0
    while len(df_next.index) > 0:
        # ------------------------------------------------------------------ # Colon
        count += 1
        if count > 90:
            break
        df_next["colon_next"] = [0]*len(df_next.index)
        df_next["colon_bin_next"] = [1]*len(df_next.index)
        # ------------------------------------------------------------------ # Heart
        df_next["heart_next"] = [0]*len(df_next.index)
        df_next["heart_bin_next"] = [1]*len(df_next.index)
        # ------------------------------------------------------------------ # Brain
        df_next["brain_next"] = [0]*len(df_next.index)
        df_next["brain_bin_next"] = [1]*len(df_next.index)
        # ------------------------------------------------------------------ # Bone
        df_next["bone_next"] = [0]*len(df_next.index)
        df_next["bone_bin_next"] = [1]*len(df_next.index)
        # ------------------------------------------------------------------ # Lung
        df_next["lung_next"] = [0]*len(df_next.index)
        df_next["lung_bin_next"] = [1]*len(df_next.index)
        # ------------------------------------------------------------------ # Esophagus
        df_next["esophagus_next"] = [0]*len(df_next.index)
        df_next["esophagus_bin_next"] = [1]*len(df_next.index)
        # ---------------------------------------------------------- #
        df_next["age_entry"] = df_next["age_exit"]
        df_next["age_exit"] = df_next["age_entry"] + 1
        #
        for col in ['colon', 'heart', 'brain', 'bone', 'lung', 'esophagus']:
            name0 = 'annual_dose_mgy_' + col + '_lag10'
            name1 = col + '_next'
            name2 = col + '_bin'
            name3 = col + '_bin_next'
            df_next[name0] = df_next[name1]
            df_next[name2] = df_next[name3]
        df_next["annual_dose_mgy_bonem_lag10"] = df_next["annual_dose_mgy_bone_lag10"]
        #
        for col in ["colon_next","heart_next","brain_next","bone_next","lung_next","esophagus_next", "colon_bin_next","heart_bin_next","brain_bin_next","bone_bin_next","lung_bin_next","esophagus_bin_next"]:
            df_next[col] = 0.0
        #
        df_next = df_next[df_next["age_exit"] < 96]
        df_gen = pd.concat([df_gen, df_next])
    df_gen.reset_index(drop = True, inplace = True)
    yob = df_gen.YOB
    df_gen["SES_HighestEd"] = pd.cut(df_gen["SES_HighestEd"], [-0.5,0.5,1.5,2.5], right = True, include_lowest = True, labels = ["Low","Medium","High"])
    # --------------------------------------------------------------------------------------------------------------------------- #
    df_gen.sort_values(['StudyID','age_entry'], ascending=[True,True],inplace=True)
    #
    for col in ['colon', 'heart', 'brain', 'bonem', 'lung', 'esophagus']:
        name0 = 'cumulative_dose_' + col + '_lag10'
        name1 = 'annual_dose_mgy_' + col + '_lag10'
        df_gen[name0] = df_gen.groupby('StudyID')[name1].cumsum() / 1000
    #
    df_gen[col_keep].to_csv("NO_EVENTS/Subj_"+str(subj_i)+"_CONST.csv")

## ---------------------------------------------- Next the Parametric models -------------------------------------------------------------------------------------- ##

## --------------------------------------------------- PROBABILITY CALCULATORS ------------------------------------------------ ##
def p0_calc(ts, ident):
    def lin_reg(x, a, b):
        return a*x + b
    def tri_fit(x, a, b, c, d):
        return a*triang.pdf(x, b, c, d)
    ts = np.array(ts)
    if ident == "00":
        poptbz = [0.0051,    -0.045]
        popthnz = [0.035]
        poptnhnz = [1.0, 0.09, 27, 60]
        poptbnz = [-0.0042223, 0.9038414]
        #
        bz = lin_reg(ts, poptbz[0], poptbz[1])
        nzh = popthnz[0]*ts/ts
        nznh = tri_fit(ts, poptnhnz[0], poptnhnz[1], poptnhnz[2], poptnhnz[3])
        bnz = lin_reg(ts, poptbnz[0], poptbnz[1])
    elif ident == "01":
        poptbz = [0.0011,     0.171]
        popthnz = [0.71]
        poptnhnz = [0.94159518, 0.09601904, 27.06720038, 58.02850693]
        poptbnz = [ 3.19703808, 0.09127464, 26.92627714, 57.02540792]
        #
        bz = lin_reg(ts, poptbz[0], poptbz[1])
        nzh = popthnz[0]*ts/ts
        nznh = tri_fit(ts, poptnhnz[0], poptnhnz[1], poptnhnz[2], poptnhnz[3])
        bnz = tri_fit(ts, poptbnz[0], poptbnz[1], poptbnz[2], poptbnz[3])
    elif ident == "10":
        poptbz = [0.0039,     0.0536]
        popthnz = [0.035]
        poptnhnz = [-0.00571706, 0.95410314]
        poptbnz = [ 4.19364951e+00, 3.82254209e-02, 2.58047853e+01, 1.14202233e+02]
        #
        bz = lin_reg(ts, poptbz[0], poptbz[1])
        nzh = popthnz[0]*ts/ts
        nznh = lin_reg(ts, poptnhnz[0], poptnhnz[1])
        bnz = tri_fit(ts, poptbnz[0], poptbnz[1], poptbnz[2], poptbnz[3])
    elif ident == "11":
        poptbz = [6.7688e-04, 8.865e-01]
        popthnz = [0.025]
        poptnhnz = [1.09762472, 0.09055709, 27.07640157, 62.08767328]
        poptbnz = [ 2.82719616, 0.10205197, 27.02117612, 57.92916286]
        #
        bz = lin_reg(ts, poptbz[0], poptbz[1])
        nzh = popthnz[0]*ts/ts
        nznh = tri_fit(ts, poptnhnz[0], poptnhnz[1], poptnhnz[2], poptnhnz[3])
        bnz = tri_fit(ts, poptbnz[0], poptbnz[1], poptbnz[2], poptbnz[3])
    else:
        assert False, "identifier wasn't coded"
    total = bz + nzh + nznh + bnz
    return [bz/total, nzh/total, nznh/total, bnz/total]

def logtrans(y, L, k, yi):
    y_dif = np.where(np.log10(y) + yi <= 0.001, 0.001, np.log10(y) + yi)
    y_dif = np.where(y_dif >= L, L-0.001, y_dif)
    return -1/k*np.log(L/y_dif-1)

def loginv(x, L, k, yi):
    return 10**(L/(1+np.exp(-k*x)) - yi)
def p1_calc(ts, ident):
    def linear_trend(x, a, b, c, d):
        return np.where(x < a, b + c*x, b+c*a - (x-a)*d)
    def double_triangle(x, a, b, c, d, e, f, g, h, i):
        return a*triang.pdf(x,b, c, d ) - e*triang.pdf(x, f, g,h)  + i
    def triangle_slope(x, a, b, c, d, e, f, g):
        return a*triang.pdf(x,b, c, d ) + e - np.where(x > f, g*(x - f), 0.0)
    ts = np.array(ts)
    #
    if ident == "01":
        popt = [9.03397625e+01, 9.07081223e-01, 1.15976818e-03, 1.27618879e-01]
        heart_trend0 = linear_trend(ts, popt[0], popt[1], popt[2], popt[3])
        cov_heart2 = 0.81*ts/ts
        return [heart_trend0, cov_heart2]
    elif ident == "10":
        popt = [9.37502922e+00, 1.69342469e-01, 2.59731002e+01, 3.71070934e+01, 5.34250177e-01, 7.49830757e+01, 6.69381730e-02]
        brain_trend =     triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        popt = [1.05068166e+01, 1.56319588e-01, 2.59730795e+01, 3.94183934e+01, 6.50576045e-01, 7.64318316e+01, 7.53003728e-02]
        lung_trend =      triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        popt = [9.91114065e+00, 1.71056883e-01, 2.59730772e+01, 3.73564910e+01, 6.69001227e-01, 7.62718801e+01, 7.52878850e-02]
        colon_trend =     triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        popt = [9.36438555e+00, 1.70529471e-01, 2.59730876e+01, 3.69182816e+01, 6.07282595e-01, 7.48729438e+01, 6.64179593e-02]
        bone_trend =      triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        popt = [8.00162220e+00, 1.87721307e-01, 2.63130033e+01, 3.27405786e+01, 6.91584912e-01, 7.44030513e+01, 6.67267639e-02]
        esophagus_trend = triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        cov_brain2 = 0.79*ts/ts
        return [brain_trend, lung_trend, colon_trend, bone_trend, esophagus_trend, cov_brain2]
    elif ident == "00":
        popt = [5.73994722, 0.18989515, 27.09976857, 28.68832209, 2.750245, 0.64036724, 71.6198358, 11.53528895, 0.77733705]
        heart_trend1 = double_triangle(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
        #
        popt = [9.37502922e+00, 1.69342469e-01, 2.59731002e+01, 3.71070934e+01, 5.34250177e-01, 7.49830757e+01, 6.69381730e-02]
        brain_trend =     triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        popt = [1.05068166e+01, 1.56319588e-01, 2.59730795e+01, 3.94183934e+01, 6.50576045e-01, 7.64318316e+01, 7.53003728e-02]
        lung_trend =      triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        popt = [9.91114065e+00, 1.71056883e-01, 2.59730772e+01, 3.73564910e+01, 6.69001227e-01, 7.62718801e+01, 7.52878850e-02]
        colon_trend =     triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        popt = [9.36438555e+00, 1.70529471e-01, 2.59730876e+01, 3.69182816e+01, 6.07282595e-01, 7.48729438e+01, 6.64179593e-02]
        bone_trend =      triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        popt = [8.00162220e+00, 1.87721307e-01, 2.63130033e+01, 3.27405786e+01, 6.91584912e-01, 7.44030513e+01, 6.67267639e-02]
        esophagus_trend = triangle_slope(ts, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        #
        cov_heart2 = 0.81*ts/ts
        cov_brain2 = 0.79*ts/ts
        cov_heartbrain = 0.69*ts/ts
        return [heart_trend1, brain_trend, lung_trend, colon_trend, bone_trend, esophagus_trend, cov_heart2, cov_brain2, cov_heartbrain]
    assert False, "p1_calc, wrong identifier"
    return
def p2_calc(ts, ident):
    def logtrap(x, a, b, c, d, e):
        return 10**(np.where(x < a, 0, np.where(x < b, (x-a)*(e-0)/(b-a), np.where(x < c, e, np.where(x < d, (x-c)*(0-e)/(d-c)+e, 0)))))
    ts = np.array(ts)
    if ident == "10":
        popt = [24.99, 30.54, 70.59, 87.94, -2.00]
        tlung  = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        popt = [25.38, 30.61, 66.81, 89.12, -2.08]
        tbrain = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        popt = [25.88, 31.23, 67.82, 88.88, -2.00]
        tcolon = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        popt = [25.38, 30.69, 68.11, 92.00, -2.00]
        tbone  = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        popt = [25.23, 30.35, 67.73, 89.88, -2.01]
        tesoph = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        return [tlung, tbrain, tcolon, tbone, tesoph]
    elif ident == "00":
        popt = [22.56, 29.58, 74.09, 88.04, -2.85]
        tlung  = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        popt = [23.70, 28.67, 71.45, 86.15, -2.91]
        tbrain = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        popt = [25.19, 30.01, 70.22, 85.71, -3.17]
        tcolon = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        popt = [23.70, 28.76, 73.93, 83.21, -2.96]
        tbone  = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        popt = [23.00, 28.00, 70.70, 84.12, -3.00]
        tesoph = logtrap(ts, popt[0], popt[1], popt[2], popt[3], popt[4])
        return [tlung, tbrain, tcolon, tbone, tesoph]
    else:
        assert False
        return
## ---------------------------------------------------------------------------------------------------------------------------- ##

df_tot = pd.read_csv("preped_data.csv")
dbins = [0]+list(np.logspace(-5,np.log10(650),40))
dbins = [float("{:.3e}".format(i)) for i in dbins]

df_dose_maxage = df_tot.groupby("StudyID").max()
df_dose_maxage = df_dose_maxage[["age_entry"]]
df_dose_maxage.columns = ["age_max"]
df_person = df_tot.join(df_dose_maxage, on = "StudyID")

subj_ids = df_person.StudyID
rng = np.random.default_rng(seed = 374221226)

N = 10


col_keep = ['StudyID', 'age_entry',
       'age_exit', 'SES_HighestEd',
       'sexm', 'yob_cat10', 'yob_cat6',
       'annual_dose_mgy_colon_lag10', 'cumulative_dose_colon_lag10',
       'annual_dose_mgy_heart_lag10', 'cumulative_dose_heart_lag10',
       'annual_dose_mgy_brain_lag10', 'cumulative_dose_brain_lag10',
       'annual_dose_mgy_bonem_lag10', 'cumulative_dose_bonem_lag10',
       'annual_dose_mgy_lung_lag10', 'cumulative_dose_lung_lag10',
       'annual_dose_mgy_esophagus_lag10', 'cumulative_dose_esophagus_lag10',
       ]

for subj_i in range(N):
    print("Dataset: " + str(subj_i))
    #
    df_person_start = df_person.copy()
    # --------------------------------------------------------------------------------------------------------------------------- #
    df_gen    = df_person_start[df_person_start["age_entry"] <= df_person_start["age_max"]].copy()
    df_next = df_person_start[df_person_start["age_entry"] == df_person_start["age_max"]].copy()
    df_next["colon_next"] = 0.0
    df_next["heart_next"] = 0.0
    df_next["brain_next"] = 0.0
    df_next["bone_next"] = 0.0
    df_next["lung_next"] = 0.0
    df_next["esophagus_next"] = 0.0
    #
    df_next.drop(['age_max'], axis = 1, inplace = True)
    df_gen.drop(['age_max'], axis = 1, inplace = True)
    count = 0
    while len(df_next.index) > 0:
        # ------------------------------------------------------------------ # prep prev
        df_next["heart_prev"] = df_next["annual_dose_mgy_heart_lag10"]
        df_next["heart_prev_zero"] = (df_next["heart_prev"] == 0)
        df_next["nonheart_prev"] = (df_next["annual_dose_mgy_brain_lag10"] + df_next["annual_dose_mgy_lung_lag10"] + df_next["annual_dose_mgy_colon_lag10"] + df_next["annual_dose_mgy_bonem_lag10"] + df_next["annual_dose_mgy_esophagus_lag10"]) / 5
        df_next["nonheart_prev_zero"] = (df_next["nonheart_prev"] == 0)
        # ---------------------------------------------------------- #
        df_next["next_ident"] = "00"
        # We start by splitting based on the previous values
        df11 = df_next[df_next["heart_prev_zero"] == True]
        df10 = df11[df11["nonheart_prev_zero"] == False]  # Heart nonzero
        df11 = df11[df11["nonheart_prev_zero"] == True] # Both nonzero
        df00 = df_next[df_next["heart_prev_zero"] == False]
        df01 = df00[df00["nonheart_prev_zero"] == True] # Nonheart nonzero
        df00 = df00[df00["nonheart_prev_zero"] == False]  # both zero
        # Now we need to determine the probabilities of each falling into each combination of zero/nonzero
        # We want to generate random values, and then assign results based on the cumulative probabilities
        # Now we assign back to df_next
        if len(df00.index) > 0:
            p00 = p0_calc(df00["age_entry"], "00")
            p_rand = np.random.rand(len(df00.index))
            res00 = np.where(p_rand < p00[0], "11", np.where(p_rand < p00[0] + p00[1], "01", np.where(p_rand < p00[0] + p00[1] +p00[2], "10", "00")))
            df00.loc[:,"next_ident"] = res00
        if len(df10.index) > 0:
            p10 = p0_calc(df10["age_entry"], "10")
            p_rand = np.random.rand(len(df10.index))
            res10 = np.where(p_rand < p10[0], "11", np.where(p_rand < p10[0] + p10[1], "01", np.where(p_rand < p10[0] + p10[1] +p10[2], "10", "00")))
            df10.loc[:,"next_ident"] = res10
        if len(df01.index) > 0:
            p01 = p0_calc(df01["age_entry"], "01")
            p_rand = np.random.rand(len(df01.index))
            res01 = np.where(p_rand < p01[0], "11", np.where(p_rand < p01[0] + p01[1], "01", np.where(p_rand < p01[0] + p01[1] +p01[2], "10", "00")))
            df01.loc[:,"next_ident"] = res01
        if len(df11.index) > 0:
            p11 = p0_calc(df11["age_entry"], "11")
            p_rand = np.random.rand(len(df11.index))
            res11 = np.where(p_rand < p11[0], "11", np.where(p_rand < p11[0] + p11[1], "01", np.where(p_rand < p11[0] + p11[1] +p11[2], "10", "00")))
            df11.loc[:,"next_ident"] = res11
        df_next = pd.concat([df11, df01, df10, df00])
        # We can split by current identifier and assign zeros where known
        df00 = df_next[df_next["next_ident"] == "00"]
        df10 = df_next[df_next["next_ident"] == "10"]
        df01 = df_next[df_next["next_ident"] == "01"]
        df11 = df_next[df_next["next_ident"] == "11"]
        # Assign known zeros
        for col in ["heart_next"]:
            df11.loc[:,col] = 0
            df10.loc[:,col] = 0
        for col in ["colon_next", "brain_next", "bone_next", "lung_next", "esophagus_next"]:
            df11.loc[:,col] = 0
            df01.loc[:,col] = 0
        # Get the remaining non-zero doses
        ##### non-zero heart
        if len(df01.index)>0:
            t01 = p1_calc(df01["age_entry"], "01")
            means = t01[0]
            covs  = t01[1]
            heart_next = [loginv(np.random.normal(means[i], np.sqrt(covs[i])), 7.0, 0.5, 4.0) for i in range(len(means))]
            df01.loc[:,"heart_next"] = heart_next
        ##### non-zero non-heart
        if len(df10.index)>0:
            t10 = p1_calc(df10["age_entry"], "10")
            brain_mean = t10[0]
            lung_mean = t10[1]
            colon_mean = t10[2]
            bone_mean = t10[3]
            esophagus_mean = t10[4]
            cov_organ2 = t10[5]
            organ_res = [loginv(np.random.multivariate_normal([brain_mean[i], lung_mean[i], colon_mean[i], bone_mean[i], esophagus_mean[i]], np.ones((5,5))*cov_organ2[i]), 7.0, 0.5, 4.0) for i in range(len(brain_mean))]
            brain_next = [i[0] for i in organ_res]
            lung_next = [i[1] for i in organ_res]
            colon_next = [i[2] for i in organ_res]
            bone_next = [i[3] for i in organ_res]
            esophagus_next = [i[4] for i in organ_res]
            # Now we need to get binomial probabilities for zeros
            p_zeros = p2_calc(df10["age_entry"], "10")
            p_lung = p_zeros[0]
            p_brain = p_zeros[1]
            p_colon = p_zeros[2]
            p_bone = p_zeros[3]
            p_esophagus = p_zeros[4]
            # Now we sample if they are zero
            zero_lung = [np.random.binomial(1, i) for i in p_lung]
            zero_brain = [np.random.binomial(1, i) for i in p_brain]
            zero_colon = [np.random.binomial(1, i) for i in p_colon]
            zero_bone = [np.random.binomial(1, i) for i in p_bone]
            zero_esophagus = [np.random.binomial(1, i) for i in p_esophagus]
            # Apply the zeros
            brain_next = np.where(zero_brain == 1, 0, brain_next)
            lung_next = np.where(zero_lung == 1, 0, lung_next)
            colon_next = np.where(zero_colon == 1, 0, colon_next)
            bone_next = np.where(zero_bone == 1, 0, bone_next)
            esophagus_next = np.where(zero_esophagus == 1, 0, esophagus_next)
            df10.loc[:,"brain_next"] = brain_next
            df10.loc[:,"lung_next"] = lung_next
            df10.loc[:,"colon_next"] = colon_next
            df10.loc[:,"bone_next"] = bone_next
            df10.loc[:,"esophagus_next"] = esophagus_next
        #### Both non-zero
        if len(df00.index)>0:
            t00 = p1_calc(df00["age_entry"], "00")
            heart_mean = t00[0]
            brain_mean = t00[1]
            lung_mean = t00[2]
            colon_mean = t00[3]
            bone_mean = t00[4]
            esophagus_mean = t00[5]
            cov_heart2 = t00[6]
            cov_organ2 = t00[7]
            cov_heartorgan = t00[8]
            organ_res = []
            for i in range(len(heart_mean)):
                mean_list = [heart_mean[i], brain_mean[i], lung_mean[i], colon_mean[i], bone_mean[i], esophagus_mean[i]]
                cov_mat = np.ones((6,6))
                cov_mat[0,0] = cov_mat[0,0] * cov_heart2[i]
                cov_mat[0,1:] = cov_mat[0,1:] * cov_heartorgan[i]
                cov_mat[1:,0] = cov_mat[1:,0] * cov_heartorgan[i]
                cov_mat[1:, 1:] = cov_mat[1:, 1:] * cov_organ2[i]
                organ_res.append(loginv(np.random.multivariate_normal(mean_list, cov_mat), 7.0, 0.5, 4.0))
            heart_next = [i[0] for i in organ_res]
            brain_next = [i[1] for i in organ_res]
            lung_next = [i[2] for i in organ_res]
            colon_next = [i[3] for i in organ_res]
            bone_next = [i[4] for i in organ_res]
            esophagus_next = [i[5] for i in organ_res]
            # Now we need to get binomial probabilities for zeros
            p_zeros = p2_calc(df00["age_entry"], "00")
            p_lung = p_zeros[0]
            p_brain = p_zeros[1]
            p_colon = p_zeros[2]
            p_bone = p_zeros[3]
            p_esophagus = p_zeros[4]
            # Now we sample if they are zero
            zero_lung = [np.random.binomial(1, i) for i in p_lung]
            zero_brain = [np.random.binomial(1, i) for i in p_brain]
            zero_colon = [np.random.binomial(1, i) for i in p_colon]
            zero_bone = [np.random.binomial(1, i) for i in p_bone]
            zero_esophagus = [np.random.binomial(1, i) for i in p_esophagus]
            # Apply the zeros
            brain_next = np.where(zero_brain == 1, 0, brain_next)
            lung_next = np.where(zero_lung == 1, 0, lung_next)
            colon_next = np.where(zero_colon == 1, 0, colon_next)
            bone_next = np.where(zero_bone == 1, 0, bone_next)
            esophagus_next = np.where(zero_esophagus == 1, 0, esophagus_next)
            #
            df00.loc[:,"heart_next"] = heart_next
            df00.loc[:,"brain_next"] = brain_next
            df00.loc[:,"lung_next"] = lung_next
            df00.loc[:,"colon_next"] = colon_next
            df00.loc[:,"bone_next"] = bone_next
            df00.loc[:,"esophagus_next"] = esophagus_next
        ## Now we bring it all together again
        df_next = pd.concat([df11, df01, df10, df00])
        # ---------------------------------------------------------- #
        df_next["age_entry"] = df_next["age_exit"]
        df_next["age_exit"] = df_next["age_entry"] + 1
        #
        for col in ['colon', 'heart', 'brain', 'bone', 'lung', 'esophagus']:
            name0 = 'annual_dose_mgy_' + col + '_lag10'
            name1 = col + '_next'
            df_next[name0] = df_next[name1]
        df_next["annual_dose_mgy_bonem_lag10"] = df_next["annual_dose_mgy_bone_lag10"]
        #
        for col in ["colon_next","heart_next","brain_next","bone_next","lung_next","esophagus_next"]:
            df_next[col] = 0.0
        #
        df_next = df_next[df_next["age_exit"] < 96]
        df_gen = pd.concat([df_gen, df_next])
    df_gen.reset_index(drop = True, inplace = True)
    yob = df_gen.YOB
    df_gen["SES_HighestEd"] = pd.cut(df_gen["SES_HighestEd"], [-0.5,0.5,1.5,2.5], right = True, include_lowest = True, labels = ["Low","Medium","High"])
    # --------------------------------------------------------------------------------------------------------------------------- #
    #
    df_gen.sort_values(['StudyID','age_entry'], ascending=[True,True],inplace=True)
    #
    for col in ['colon', 'heart', 'brain', 'bonem', 'lung', 'esophagus']:
        name0 = 'cumulative_dose_' + col + '_lag10'
        name1 = 'annual_dose_mgy_' + col + '_lag10'
        df_gen[name0] = df_gen.groupby('StudyID')[name1].cumsum() / 1000
    #
    df_gen[col_keep].to_csv("NO_EVENTS/Subj_"+str(subj_i)+"_PARA.csv")

## ---------------------------------------------- Finally the Random Forest models -------------------------------------------------------------------------------------- ##
df_tot = pd.read_csv("preped_data.csv")

# -------------------------------------------------------------------------------------- #

# Colon
x_col_colon = ['year', 'YOB', 'SES_HighestEd', 'sexm', 'YOH', 'age_entry', 'age_exit',
              'colon_bin', 'heart_bin', 'brain_bin', 'bone_bin', 'lung_bin', 'esophagus_bin']
y_col_colon = ['colon_bin_next']
X = df_tot[x_col_colon]
y = np.ravel(df_tot[y_col_colon].astype(str))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
base_rf_colon = RandomForestClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=7,
                                 n_estimators=20, n_jobs=-1, random_state = 0)
base_rf_colon.fit(X_train, y_train)
print("colon train")
# Heart
x_col_heart = ['year', 'YOB', 'SES_HighestEd', 'sexm', 'YOH', 'age_entry', 'age_exit',
              'colon_bin', 'heart_bin', 'brain_bin', 'bone_bin', 'lung_bin', 'esophagus_bin',
              'colon_bin_next']
y_col_heart = ['heart_bin_next']
X = df_tot[x_col_heart]
y = np.ravel(df_tot[y_col_heart].astype(str))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
base_rf_heart = RandomForestClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=7,
                                 n_estimators=20, n_jobs=-1, random_state = 0)
base_rf_heart.fit(X_train, y_train)
print("heart train")
# Brain
x_col_brain = ['year', 'YOB', 'SES_HighestEd', 'sexm', 'YOH', 'age_entry', 'age_exit',
              'colon_bin', 'heart_bin', 'brain_bin', 'bone_bin', 'lung_bin', 'esophagus_bin',
              'colon_bin_next', 'heart_bin_next']
y_col_brain = ['brain_bin_next']
X = df_tot[x_col_brain]
y = np.ravel(df_tot[y_col_brain].astype(str))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
base_rf_brain = RandomForestClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=7,
                                 n_estimators=20, n_jobs=-1, random_state = 0)
base_rf_brain.fit(X_train, y_train)
print("brain train")
# Bone
x_col_bone = ['year', 'YOB', 'SES_HighestEd', 'sexm', 'YOH', 'age_entry', 'age_exit',
              'colon_bin', 'heart_bin', 'brain_bin', 'bone_bin', 'lung_bin', 'esophagus_bin',
              'colon_bin_next', 'heart_bin_next', 'brain_bin_next']
y_col_bone = ['bone_bin_next']
X = df_tot[x_col_bone]
y = np.ravel(df_tot[y_col_bone].astype(str))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
base_rf_bone = RandomForestClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=7,
                                 n_estimators=20, n_jobs=-1, random_state = 0)
base_rf_bone.fit(X_train, y_train)
print("bone train")
# Lung
x_col_lung = ['year', 'YOB', 'SES_HighestEd', 'sexm', 'YOH', 'age_entry', 'age_exit',
              'colon_bin', 'heart_bin', 'brain_bin', 'bone_bin', 'lung_bin', 'esophagus_bin',
              'colon_bin_next', 'heart_bin_next', 'brain_bin_next', 'bone_bin_next']
y_col_lung = ['lung_bin_next']
X = df_tot[x_col_lung]
y = np.ravel(df_tot[y_col_lung].astype(str))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
base_rf_lung = RandomForestClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=7,
                                 n_estimators=20, n_jobs=-1, random_state = 0)
base_rf_lung.fit(X_train, y_train)
print("lung train")
# Esophagus
x_col_eso = ['year', 'YOB', 'SES_HighestEd', 'sexm', 'YOH', 'age_entry', 'age_exit',
              'colon_bin', 'heart_bin', 'brain_bin', 'bone_bin', 'lung_bin', 'esophagus_bin',
              'colon_bin_next', 'heart_bin_next', 'brain_bin_next', 'bone_bin_next', 'lung_bin_next']
y_col_eso = ['esophagus_bin_next']
X = df_tot[x_col_eso]
y = np.ravel(df_tot[y_col_eso].astype(str))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
base_rf_eso = RandomForestClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=7,
                                 n_estimators=20, n_jobs=-1, random_state = 0)
base_rf_eso.fit(X_train, y_train)
print("esophagus train")

colon_classes = base_rf_colon.classes_
heart_classes = base_rf_heart.classes_
brain_classes = base_rf_brain.classes_
bone_classes  = base_rf_bone.classes_
lung_classes  = base_rf_lung.classes_
eso_classes   = base_rf_eso.classes_


dbins = [0]+list(np.logspace(-5,np.log10(650),40))
dbins = [float("{:.3e}".format(i)) for i in dbins]


df_dose_maxage = df_tot.groupby("StudyID").max()
df_dose_maxage = df_dose_maxage[["age_entry"]]
df_dose_maxage.columns = ["age_max"]
df_person = df_tot.join(df_dose_maxage, on = "StudyID")

subj_ids = df_person.StudyID
rng = np.random.default_rng(seed = 374221226)
#
#m = 100000
N = 10


col_keep = ['StudyID', 'age_entry',
       'age_exit', 'SES_HighestEd',
       'sexm', 'yob_cat10', 'yob_cat6',
       'annual_dose_mgy_colon_lag10', 'cumulative_dose_colon_lag10',
       'annual_dose_mgy_heart_lag10', 'cumulative_dose_heart_lag10',
       'annual_dose_mgy_brain_lag10', 'cumulative_dose_brain_lag10',
       'annual_dose_mgy_bonem_lag10', 'cumulative_dose_bonem_lag10',
       'annual_dose_mgy_lung_lag10', 'cumulative_dose_lung_lag10',
       'annual_dose_mgy_esophagus_lag10', 'cumulative_dose_esophagus_lag10',
       ]

for subj_i in range(N):
    print("Dataset: " + str(subj_i))
    #
    df_person_start = df_person.copy()
    # --------------------------------------------------------------------------------------------------------------------------- #
    df_gen    = df_person_start[df_person_start["age_entry"] <= df_person_start["age_max"]].copy()
    df_next = df_person_start[df_person_start["age_entry"] == df_person_start["age_max"]].copy()
    df_next["colon_next"] = 0.0
    df_next["heart_next"] = 0.0
    df_next["brain_next"] = 0.0
    df_next["bone_next"] = 0.0
    df_next["lung_next"] = 0.0
    df_next["esophagus_next"] = 0.0
    #
    df_next["colon_bin_next"] = 0.0
    df_next["heart_bin_next"] = 0.0
    df_next["brain_bin_next"] = 0.0
    df_next["bone_bin_next"] = 0.0
    df_next["lung_bin_next"] = 0.0
    df_next["esophagus_bin_next"] = 0.0
    #
    df_next.drop(['age_max'], axis = 1, inplace = True)
    df_gen.drop(['age_max'], axis = 1, inplace = True)
    #df_next = df_gen.copy()
    count = 0
    while len(df_next.index) > 0:
        # ------------------------------------------------------------------ # Colon
        X_next_colon = df_next[x_col_colon]
        count += 1
        if count > 90:
            break
        y_next_colon_dist = base_rf_colon.predict_proba(X_next_colon)

        y_next_colon_bin = []
        y_next_colon = []
        for dist in y_next_colon_dist:
            cum_dist = np.cumsum(dist)
            U = rng.random()
            true_bin = 0
            for i in range(len(cum_dist)):
                if U <= cum_dist[i]:
                    true_bin = int(colon_classes[i])
                    break
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_colon_bin.append(true_bin)
            y_next_colon.append(dose)
        df_next["colon_next"] = y_next_colon
        df_next["colon_bin_next"] = y_next_colon_bin
        # ------------------------------------------------------------------ # Heart
        X_next_heart = df_next[x_col_heart]

        y_next_heart_dist = base_rf_heart.predict_proba(X_next_heart)

        y_next_heart_bin = []
        y_next_heart = []
        for dist in y_next_heart_dist:
            cum_dist = np.cumsum(dist)
            U = rng.random()
            true_bin = 0
            for i in range(len(cum_dist)):
                if U <= cum_dist[i]:
                    true_bin = int(heart_classes[i])
                    break
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_heart_bin.append(true_bin)
            y_next_heart.append(dose)
        df_next["heart_next"] = y_next_heart
        df_next["heart_bin_next"] = y_next_heart_bin
        # ------------------------------------------------------------------ # Brain
        X_next_brain = df_next[x_col_brain]

        y_next_brain_dist = base_rf_brain.predict_proba(X_next_brain)

        y_next_brain_bin = []
        y_next_brain = []
        for dist in y_next_brain_dist:
            cum_dist = np.cumsum(dist)
            U = rng.random()
            true_bin = 0
            for i in range(len(cum_dist)):
                if U <= cum_dist[i]:
                    true_bin = int(brain_classes[i])
                    break
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_brain_bin.append(true_bin)
            y_next_brain.append(dose)
        df_next["brain_next"] = y_next_brain
        df_next["brain_bin_next"] = y_next_brain_bin
        # ------------------------------------------------------------------ # Bone
        X_next_bone = df_next[x_col_bone]

        y_next_bone_dist = base_rf_bone.predict_proba(X_next_bone)

        y_next_bone_bin = []
        y_next_bone = []
        for dist in y_next_bone_dist:
            cum_dist = np.cumsum(dist)
            U = rng.random()
            true_bin = 0
            for i in range(len(cum_dist)):
                if U <= cum_dist[i]:
                    true_bin = int(bone_classes[i])
                    break
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_bone_bin.append(true_bin)
            y_next_bone.append(dose)
        df_next["bone_next"] = y_next_bone
        df_next["bone_bin_next"] = y_next_bone_bin
        # ------------------------------------------------------------------ # Lung
        X_next_lung = df_next[x_col_lung]

        y_next_lung_dist = base_rf_lung.predict_proba(X_next_lung)

        y_next_lung_bin = []
        y_next_lung = []
        for dist in y_next_lung_dist:
            cum_dist = np.cumsum(dist)
            U = rng.random()
            true_bin = 0
            for i in range(len(cum_dist)):
                if U <= cum_dist[i]:
                    true_bin = int(lung_classes[i])
                    break
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_lung_bin.append(true_bin)
            y_next_lung.append(dose)
        df_next["lung_next"] = y_next_lung
        df_next["lung_bin_next"] = y_next_lung_bin
        # ------------------------------------------------------------------ # Esophagus
        X_next_eso = df_next[x_col_eso]

        y_next_eso_dist = base_rf_eso.predict_proba(X_next_eso)

        y_next_eso_bin = []
        y_next_eso = []
        for dist in y_next_eso_dist:
            cum_dist = np.cumsum(dist)
            U = rng.random()
            true_bin = 0
            for i in range(len(cum_dist)):
                if U <= cum_dist[i]:
                    true_bin = int(eso_classes[i])
                    break
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_eso_bin.append(true_bin)
            y_next_eso.append(dose)
        df_next["esophagus_next"] = y_next_eso
        df_next["esophagus_bin_next"] = y_next_eso_bin
        # ---------------------------------------------------------- #
        df_next["age_entry"] = df_next["age_exit"]
        df_next["age_exit"] = df_next["age_entry"] + 1
        #
        for col in ['colon', 'heart', 'brain', 'bone', 'lung', 'esophagus']:
            name0 = 'annual_dose_mgy_' + col + '_lag10'
            name1 = col + '_next'
            name2 = col + '_bin'
            name3 = col + '_bin_next'
            df_next[name0] = df_next[name1]
            df_next[name2] = df_next[name3]
        df_next["annual_dose_mgy_bonem_lag10"] = df_next["annual_dose_mgy_bone_lag10"]
        #
        for col in ["colon_next","heart_next","brain_next","bone_next","lung_next","esophagus_next", "colon_bin_next","heart_bin_next","brain_bin_next","bone_bin_next","lung_bin_next","esophagus_bin_next"]:
            df_next[col] = 0.0
        #
        df_next = df_next[df_next["age_exit"] < 96]
        df_gen = pd.concat([df_gen, df_next])
    df_gen.reset_index(drop = True, inplace = True)
    yob = df_gen.YOB
    df_gen["SES_HighestEd"] = pd.cut(df_gen["SES_HighestEd"], [-0.5,0.5,1.5,2.5], right = True, include_lowest = True, labels = ["Low","Medium","High"])
    # --------------------------------------------------------------------------------------------------------------------------- #
    df_gen.sort_values(['StudyID','age_entry'], ascending=[True,True],inplace=True)
    #
    for col in ['colon', 'heart', 'brain', 'bonem', 'lung', 'esophagus']:
        name0 = 'cumulative_dose_' + col + '_lag10'
        name1 = 'annual_dose_mgy_' + col + '_lag10'
        df_gen[name0] = df_gen.groupby('StudyID')[name1].cumsum() / 1000
    #
    df_gen[col_keep].to_csv("NO_EVENTS/Subj_"+str(subj_i)+"_RF.csv")

## ------------------------------------------------------------------- And the Random Forest Directly Predicting doses ------------------------ ##

rng = np.random.default_rng(seed = 374221226)
#
N = 10


col_keep = ['StudyID', 'age_entry',
       'age_exit', 'SES_HighestEd',
       'sexm', 'yob_cat10', 'yob_cat6',
       'annual_dose_mgy_colon_lag10', 'cumulative_dose_colon_lag10',
       'annual_dose_mgy_heart_lag10', 'cumulative_dose_heart_lag10',
       'annual_dose_mgy_brain_lag10', 'cumulative_dose_brain_lag10',
       'annual_dose_mgy_bonem_lag10', 'cumulative_dose_bonem_lag10',
       'annual_dose_mgy_lung_lag10', 'cumulative_dose_lung_lag10',
       'annual_dose_mgy_esophagus_lag10', 'cumulative_dose_esophagus_lag10',
       ]

for subj_i in range(N):
    print("Dataset: " + str(subj_i))
    #
    df_person_start = df_person.copy()
    # --------------------------------------------------------------------------------------------------------------------------- #

    df_gen    = df_person_start[df_person_start["age_entry"] <= df_person_start["age_max"]].copy()
    df_next = df_person_start[df_person_start["age_entry"] == df_person_start["age_max"]].copy()
    df_next["colon_next"] = 0.0
    df_next["heart_next"] = 0.0
    df_next["brain_next"] = 0.0
    df_next["bone_next"] = 0.0
    df_next["lung_next"] = 0.0
    df_next["esophagus_next"] = 0.0
    #
    df_next["colon_bin_next"] = 0.0
    df_next["heart_bin_next"] = 0.0
    df_next["brain_bin_next"] = 0.0
    df_next["bone_bin_next"] = 0.0
    df_next["lung_bin_next"] = 0.0
    df_next["esophagus_bin_next"] = 0.0
    #
    df_next.drop(['age_max'], axis = 1, inplace = True)
    df_gen.drop(['age_max'], axis = 1, inplace = True)
    count = 0
    while len(df_next.index) > 0:
        # ------------------------------------------------------------------ # Colon
        X_next_colon = df_next[x_col_colon]
        count += 1
        if count > 90:
            break
        y_next_colon_bin = base_rf_colon.predict(X_next_colon)
        y_next_colon = []
        for true_bin in y_next_colon_bin:
            true_bin = int(true_bin)
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_colon.append(dose)
        df_next["colon_next"] = y_next_colon
        df_next["colon_bin_next"] = y_next_colon_bin
        # ------------------------------------------------------------------ # Heart
        X_next_heart = df_next[x_col_heart]
        y_next_heart_bin = base_rf_heart.predict(X_next_heart)
        y_next_heart = []
        for true_bin in y_next_heart_bin:
            true_bin = int(true_bin)
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_heart.append(dose)
        df_next["heart_next"] = y_next_heart
        df_next["heart_bin_next"] = y_next_heart_bin
        # ------------------------------------------------------------------ # Brain
        X_next_brain = df_next[x_col_brain]
        y_next_brain_bin = base_rf_brain.predict(X_next_brain)
        y_next_brain = []
        for true_bin in y_next_brain_bin:
            true_bin = int(true_bin)
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_brain.append(dose)
        df_next["brain_next"] = y_next_brain
        df_next["brain_bin_next"] = y_next_brain_bin
        # ------------------------------------------------------------------ # Bone
        X_next_bone = df_next[x_col_bone]
        y_next_bone_bin = base_rf_bone.predict(X_next_bone)
        y_next_bone = []
        for true_bin in y_next_bone_bin:
            true_bin = int(true_bin)
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_bone.append(dose)
        df_next["bone_next"] = y_next_bone
        df_next["bone_bin_next"] = y_next_bone_bin
        # ------------------------------------------------------------------ # Lung
        X_next_lung = df_next[x_col_lung]
        y_next_lung_bin = base_rf_lung.predict(X_next_lung)
        y_next_lung = []
        for true_bin in y_next_lung_bin:
            true_bin = int(true_bin)
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_lung.append(dose)
        df_next["lung_next"] = y_next_lung
        df_next["lung_bin_next"] = y_next_lung_bin
        # ------------------------------------------------------------------ # Esophagus
        X_next_eso = df_next[x_col_eso]
        y_next_eso_bin = base_rf_eso.predict(X_next_eso)
        y_next_eso = []
        for true_bin in y_next_eso_bin:
            true_bin = int(true_bin)
            if true_bin <= 1:
                dose = 0.0;
            elif true_bin >= len(dbins):
                dose = dbins[-1]
            else:
                bin_low = dbins[true_bin-1]
                bin_high = dbins[true_bin]
                U = rng.random()
                dose = (bin_high - bin_low)*U + bin_low
            #
            y_next_eso.append(dose)
        df_next["esophagus_next"] = y_next_eso
        df_next["esophagus_bin_next"] = y_next_eso_bin
        # ---------------------------------------------------------- #
        df_next["age_entry"] = df_next["age_exit"]
        df_next["age_exit"] = df_next["age_entry"] + 1
        #
        for col in ['colon', 'heart', 'brain', 'bone', 'lung', 'esophagus']:
            name0 = 'annual_dose_mgy_' + col + '_lag10'
            name1 = col + '_next'
            name2 = col + '_bin'
            name3 = col + '_bin_next'
            df_next[name0] = df_next[name1]
            df_next[name2] = df_next[name3]
        df_next["annual_dose_mgy_bonem_lag10"] = df_next["annual_dose_mgy_bone_lag10"]
        #
        for col in ["colon_next","heart_next","brain_next","bone_next","lung_next","esophagus_next", "colon_bin_next","heart_bin_next","brain_bin_next","bone_bin_next","lung_bin_next","esophagus_bin_next"]:
            df_next[col] = 0.0
        #
        df_next = df_next[df_next["age_exit"] < 96]
        df_gen = pd.concat([df_gen, df_next])
    df_gen.reset_index(drop = True, inplace = True)
    yob = df_gen.YOB
    df_gen["SES_HighestEd"] = pd.cut(df_gen["SES_HighestEd"], [-0.5,0.5,1.5,2.5], right = True, include_lowest = True, labels = ["Low","Medium","High"])
    # --------------------------------------------------------------------------------------------------------------------------- #
    #
    df_gen.sort_values(['StudyID','age_entry'], ascending=[True,True],inplace=True)
    #
    for col in ['colon', 'heart', 'brain', 'bonem', 'lung', 'esophagus']:
        name0 = 'cumulative_dose_' + col + '_lag10'
        name1 = 'annual_dose_mgy_' + col + '_lag10'
        df_gen[name0] = df_gen.groupby('StudyID')[name1].cumsum() / 1000
    #
    df_gen[col_keep].to_csv("NO_EVENTS/Subj_"+str(subj_i)+"_RF_CATEG.csv")
    
'''
To calculate the accuracy plots, the loops were updated so that 'df_next' was replaced with 'df_person_start', 'df_gen' was removed, and the 'while len(df_next.index) > 0' loop was removed
Instead of assigning to 'organ_next', the guesses are assigned to 'organ_next_guess', and no cumulative doses are calculated.
As an example, the parametric and random forest versions are reproduced below. For readability, the random forest version mode prediction method is not reproduced.
'''

df_tot = pd.read_csv("preped_data.csv")
# -------------------------------------------------------------------------------------- #

dbins = [0]+list(np.logspace(-5,np.log10(650),40))
dbins = [float("{:.3e}".format(i)) for i in dbins]


df_dose_maxage = df_tot.groupby("StudyID").max()
df_dose_maxage = df_dose_maxage[["age_entry"]]
df_dose_maxage.columns = ["age_max"]
df_person = df_tot.join(df_dose_maxage, on = "StudyID")

person_train, person_test = train_test_split(df_person, test_size=0.20, random_state=0)

subj_ids = person_test.StudyID
rng = np.random.default_rng(seed = 374221226)
#
N = 10


col_keep = ['StudyID', 'age_entry',
       'age_exit', 'SES_HighestEd',
       'sexm', 'yob_cat10', 'yob_cat6',
       'annual_dose_mgy_colon_lag10', 'cumulative_dose_colon_lag10',
       'annual_dose_mgy_heart_lag10', 'cumulative_dose_heart_lag10',
       'annual_dose_mgy_brain_lag10', 'cumulative_dose_brain_lag10',
       'annual_dose_mgy_bonem_lag10', 'cumulative_dose_bonem_lag10',
       'annual_dose_mgy_lung_lag10', 'cumulative_dose_lung_lag10',
       'annual_dose_mgy_esophagus_lag10', 'cumulative_dose_esophagus_lag10',
       ]

for subj_i in range(N):
    #
    df_person_start = person_test.copy()
    # ------------------------------------------------------------------ # prep prev
    df_person_start["heart_prev"] = df_person_start["annual_dose_mgy_heart_lag10"]
    df_person_start["heart_prev_zero"] = (df_person_start["heart_prev"] == 0)
    df_person_start["nonheart_prev"] = (df_person_start["annual_dose_mgy_brain_lag10"] + df_person_start["annual_dose_mgy_lung_lag10"] + df_person_start["annual_dose_mgy_colon_lag10"] + df_person_start["annual_dose_mgy_bonem_lag10"] + df_person_start["annual_dose_mgy_esophagus_lag10"]) / 5
    df_person_start["nonheart_prev_zero"] = (df_person_start["nonheart_prev"] == 0)
    # ---------------------------------------------------------- #
    df_person_start["next_ident"] = "00"
    # We start by splitting based on the previous values
    df11 = df_person_start[df_person_start["heart_prev_zero"] == True].copy()
    df10 = df11[df11["nonheart_prev_zero"] == False].copy()  # Heart nonzero
    df11 = df11[df11["nonheart_prev_zero"] == True].copy() # Both nonzero
    df00 = df_person_start[df_person_start["heart_prev_zero"] == False].copy()
    df01 = df00[df00["nonheart_prev_zero"] == True].copy() # Nonheart nonzero
    df00 = df00[df00["nonheart_prev_zero"] == False].copy()  # both zero
    # Now we need to determine the probabilities of each falling into each combination of zero/nonzero
    # We want to generate random values, and then assign results based on the cumulative probabilities
    # Now we assign back to df_next
    if len(df00.index) > 0:
        p00 = p0_calc(df00["age_entry"], "00")
        p_rand = np.random.rand(len(df00.index))
        res00 = np.where(p_rand < p00[0], "11", np.where(p_rand < p00[0] + p00[1], "01", np.where(p_rand < p00[0] + p00[1] +p00[2], "10", "00")))
        df00.loc[:,"next_ident"] = res00
    if len(df10.index) > 0:
        p10 = p0_calc(df10["age_entry"], "10")
        p_rand = np.random.rand(len(df10.index))
        res10 = np.where(p_rand < p10[0], "11", np.where(p_rand < p10[0] + p10[1], "01", np.where(p_rand < p10[0] + p10[1] +p10[2], "10", "00")))
        df10.loc[:,"next_ident"] = res10
    if len(df01.index) > 0:
        p01 = p0_calc(df01["age_entry"], "01")
        p_rand = np.random.rand(len(df01.index))
        res01 = np.where(p_rand < p01[0], "11", np.where(p_rand < p01[0] + p01[1], "01", np.where(p_rand < p01[0] + p01[1] +p01[2], "10", "00")))
        df01.loc[:,"next_ident"] = res01
    if len(df11.index) > 0:
        p11 = p0_calc(df11["age_entry"], "11")
        p_rand = np.random.rand(len(df11.index))
        res11 = np.where(p_rand < p11[0], "11", np.where(p_rand < p11[0] + p11[1], "01", np.where(p_rand < p11[0] + p11[1] +p11[2], "10", "00")))
        df11.loc[:,"next_ident"] = res11
    df_person_start = pd.concat([df11, df01, df10, df00])
    # We can split by current identifier and assign zeros where known
    df00 = df_person_start[df_person_start["next_ident"] == "00"].copy()
    df10 = df_person_start[df_person_start["next_ident"] == "10"].copy()
    df01 = df_person_start[df_person_start["next_ident"] == "01"].copy()
    df11 = df_person_start[df_person_start["next_ident"] == "11"].copy()
    # Assign known zeros
    for col in ["heart_next_guess"]:
        df11.loc[:,col] = 0
        df10.loc[:,col] = 0
    for col in ["colon_next_guess", "brain_next_guess", "bone_next_guess", "lung_next_guess", "esophagus_next_guess"]:
        df11.loc[:,col] = 0
        df01.loc[:,col] = 0
    # Get the remaining non-zero doses
    ##### non-zero heart
    if len(df01.index)>0:
        t01 = p1_calc(df01["age_entry"], "01")
        means = t01[0]
        covs  = t01[1]
        heart_next = [loginv(np.random.normal(means[i], np.sqrt(covs[i])), 7.0, 0.5, 4.0) for i in range(len(means))]
        df01.loc[:,"heart_next_guess"] = heart_next
    ##### non-zero non-heart
    if len(df10.index)>0:
        t10 = p1_calc(df10["age_entry"], "10")
        brain_mean = t10[0]
        lung_mean = t10[1]
        colon_mean = t10[2]
        bone_mean = t10[3]
        esophagus_mean = t10[4]
        cov_organ2 = t10[5]
        organ_res = [loginv(np.random.multivariate_normal([brain_mean[i], lung_mean[i], colon_mean[i], bone_mean[i], esophagus_mean[i]], np.ones((5,5))*cov_organ2[i]), 7.0, 0.5, 4.0) for i in range(len(brain_mean))]
        brain_next = [i[0] for i in organ_res]
        lung_next = [i[1] for i in organ_res]
        colon_next = [i[2] for i in organ_res]
        bone_next = [i[3] for i in organ_res]
        esophagus_next = [i[4] for i in organ_res]
        # Now we need to get binomial probabilities for zeros
        p_zeros = p2_calc(df10["age_entry"], "10")
        p_lung = p_zeros[0]
        p_brain = p_zeros[1]
        p_colon = p_zeros[2]
        p_bone = p_zeros[3]
        p_esophagus = p_zeros[4]
        # Now we sample if they are zero
        zero_lung = [np.random.binomial(1, i) for i in p_lung]
        zero_brain = [np.random.binomial(1, i) for i in p_brain]
        zero_colon = [np.random.binomial(1, i) for i in p_colon]
        zero_bone = [np.random.binomial(1, i) for i in p_bone]
        zero_esophagus = [np.random.binomial(1, i) for i in p_esophagus]
        # Apply the zeros
        brain_next = np.where(zero_brain == 1, 0, brain_next)
        lung_next = np.where(zero_lung == 1, 0, lung_next)
        colon_next = np.where(zero_colon == 1, 0, colon_next)
        bone_next = np.where(zero_bone == 1, 0, bone_next)
        esophagus_next = np.where(zero_esophagus == 1, 0, esophagus_next)
        df10.loc[:,"brain_next_guess"] = brain_next
        df10.loc[:,"lung_next_guess"] = lung_next
        df10.loc[:,"colon_next_guess"] = colon_next
        df10.loc[:,"bone_next_guess"] = bone_next
        df10.loc[:,"esophagus_next_guess"] = esophagus_next
    #### Both non-zero
    if len(df00.index)>0:
        t00 = p1_calc(df00["age_entry"], "00")
        heart_mean = t00[0]
        brain_mean = t00[1]
        lung_mean = t00[2]
        colon_mean = t00[3]
        bone_mean = t00[4]
        esophagus_mean = t00[5]
        cov_heart2 = t00[6]
        cov_organ2 = t00[7]
        cov_heartorgan = t00[8]
        organ_res = []
        for i in range(len(heart_mean)):
            mean_list = [heart_mean[i], brain_mean[i], lung_mean[i], colon_mean[i], bone_mean[i], esophagus_mean[i]]
            cov_mat = np.ones((6,6))
            cov_mat[0,0] = cov_mat[0,0] * cov_heart2[i]
            cov_mat[0,1:] = cov_mat[0,1:] * cov_heartorgan[i]
            cov_mat[1:,0] = cov_mat[1:,0] * cov_heartorgan[i]
            cov_mat[1:, 1:] = cov_mat[1:, 1:] * cov_organ2[i]
            organ_res.append(loginv(np.random.multivariate_normal(mean_list, cov_mat), 7.0, 0.5, 4.0))
        heart_next = [i[0] for i in organ_res]
        brain_next = [i[1] for i in organ_res]
        lung_next = [i[2] for i in organ_res]
        colon_next = [i[3] for i in organ_res]
        bone_next = [i[4] for i in organ_res]
        esophagus_next = [i[5] for i in organ_res]
        # Now we need to get binomial probabilities for zeros
        p_zeros = p2_calc(df00["age_entry"], "00")
        p_lung = p_zeros[0]
        p_brain = p_zeros[1]
        p_colon = p_zeros[2]
        p_bone = p_zeros[3]
        p_esophagus = p_zeros[4]
        # Now we sample if they are zero
        zero_lung = [np.random.binomial(1, i) for i in p_lung]
        zero_brain = [np.random.binomial(1, i) for i in p_brain]
        zero_colon = [np.random.binomial(1, i) for i in p_colon]
        zero_bone = [np.random.binomial(1, i) for i in p_bone]
        zero_esophagus = [np.random.binomial(1, i) for i in p_esophagus]
        # Apply the zeros
        brain_next = np.where(zero_brain == 1, 0, brain_next)
        lung_next = np.where(zero_lung == 1, 0, lung_next)
        colon_next = np.where(zero_colon == 1, 0, colon_next)
        bone_next = np.where(zero_bone == 1, 0, bone_next)
        esophagus_next = np.where(zero_esophagus == 1, 0, esophagus_next)
        #
        df00.loc[:,"heart_next_guess"] = heart_next
        df00.loc[:,"brain_next_guess"] = brain_next
        df00.loc[:,"lung_next_guess"] = lung_next
        df00.loc[:,"colon_next_guess"] = colon_next
        df00.loc[:,"bone_next_guess"] = bone_next
        df00.loc[:,"esophagus_next_guess"] = esophagus_next
    ## Now we bring it all together again
    df_person_start = pd.concat([df11, df01, df10, df00])
    # ---------------------------------------------------------- #
    df_person_start.to_csv("ACCURACY/Subj_PARA_"+str(subj_i)+"_accuracy.csv")

## --------------------------- Random Forest ------------------------------ ##
person_train, person_test = train_test_split(df_person, test_size=0.20, random_state=0)

subj_ids = person_test.StudyID
rng = np.random.default_rng(seed = 374221226)
#
#m = 100000
N = 10


col_keep = ['StudyID', 'age_entry',
       'age_exit', 'SES_HighestEd',
       'sexm', 'yob_cat10', 'yob_cat6',
       'annual_dose_mgy_colon_lag10', 'cumulative_dose_colon_lag10',
       'annual_dose_mgy_heart_lag10', 'cumulative_dose_heart_lag10',
       'annual_dose_mgy_brain_lag10', 'cumulative_dose_brain_lag10',
       'annual_dose_mgy_bonem_lag10', 'cumulative_dose_bonem_lag10',
       'annual_dose_mgy_lung_lag10', 'cumulative_dose_lung_lag10',
       'annual_dose_mgy_esophagus_lag10', 'cumulative_dose_esophagus_lag10',
       ]

for subj_i in range(N):
    print("Dataset: " + str(subj_i))
    #
    df_person_start = person_test.copy()
    # --------------------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------ # Colon
    X_next_colon = df_person_start[x_col_colon]
    y_next_colon_dist = base_rf_colon.predict_proba(X_next_colon)

    y_next_colon_bin = []
    y_next_colon = []
    for dist in y_next_colon_dist:
        cum_dist = np.cumsum(dist)
        U = rng.random()
        true_bin = 0
        for i in range(len(cum_dist)):
            if U <= cum_dist[i]:
                true_bin = int(colon_classes[i])
                break
        if true_bin <= 1:
            dose = 0.0;
        elif true_bin >= len(dbins):
                dose = dbins[-1]
        else:
            bin_low = dbins[true_bin-1]
            bin_high = dbins[true_bin]
            U = rng.random()
            dose = (bin_high - bin_low)*U + bin_low
        #
        y_next_colon_bin.append(true_bin)
        y_next_colon.append(dose)
    df_person_start["colon_next_guess"] = y_next_colon
    df_person_start["colon_bin_next"] = y_next_colon_bin
    # ------------------------------------------------------------------ # Heart
    X_next_heart = df_person_start[x_col_heart]

    y_next_heart_dist = base_rf_heart.predict_proba(X_next_heart)

    y_next_heart_bin = []
    y_next_heart = []
    for dist in y_next_heart_dist:
        cum_dist = np.cumsum(dist)
        U = rng.random()
        true_bin = 0
        for i in range(len(cum_dist)):
            if U <= cum_dist[i]:
                true_bin = int(heart_classes[i])
                break
        if true_bin <= 1:
            dose = 0.0;
        elif true_bin >= len(dbins):
            dose = dbins[-1]
        else:
            bin_low = dbins[true_bin-1]
            bin_high = dbins[true_bin]
            U = rng.random()
            dose = (bin_high - bin_low)*U + bin_low
        #
        y_next_heart_bin.append(true_bin)
        y_next_heart.append(dose)
    df_person_start["heart_next_guess"] = y_next_heart
    df_person_start["heart_bin_next"] = y_next_heart_bin
    # ------------------------------------------------------------------ # Brain
    X_next_brain = df_person_start[x_col_brain]

    y_next_brain_dist = base_rf_brain.predict_proba(X_next_brain)

    y_next_brain_bin = []
    y_next_brain = []
    for dist in y_next_brain_dist:
        cum_dist = np.cumsum(dist)
        U = rng.random()
        true_bin = 0
        for i in range(len(cum_dist)):
            if U <= cum_dist[i]:
                true_bin = int(brain_classes[i])
                break
        if true_bin <= 1:
            dose = 0.0;
        elif true_bin >= len(dbins):
            dose = dbins[-1]
        else:
            bin_low = dbins[true_bin-1]
            bin_high = dbins[true_bin]
            U = rng.random()
            dose = (bin_high - bin_low)*U + bin_low
        #
        y_next_brain_bin.append(true_bin)
        y_next_brain.append(dose)
    df_person_start["brain_next_guess"] = y_next_brain
    df_person_start["brain_bin_next"] = y_next_brain_bin
    # ------------------------------------------------------------------ # Bone
    X_next_bone = df_person_start[x_col_bone]

    y_next_bone_dist = base_rf_bone.predict_proba(X_next_bone)

    y_next_bone_bin = []
    y_next_bone = []
    for dist in y_next_bone_dist:
        cum_dist = np.cumsum(dist)
        U = rng.random()
        true_bin = 0
        for i in range(len(cum_dist)):
            if U <= cum_dist[i]:
                true_bin = int(bone_classes[i])
                break
        if true_bin <= 1:
            dose = 0.0;
        elif true_bin >= len(dbins):
            dose = dbins[-1]
        else:
            bin_low = dbins[true_bin-1]
            bin_high = dbins[true_bin]
            U = rng.random()
            dose = (bin_high - bin_low)*U + bin_low
        #
        y_next_bone_bin.append(true_bin)
        y_next_bone.append(dose)
    df_person_start["bone_next_guess"] = y_next_bone
    df_person_start["bone_bin_next"] = y_next_bone_bin
    # ------------------------------------------------------------------ # Lung
    X_next_lung = df_person_start[x_col_lung]

    y_next_lung_dist = base_rf_lung.predict_proba(X_next_lung)

    y_next_lung_bin = []
    y_next_lung = []
    for dist in y_next_lung_dist:
        cum_dist = np.cumsum(dist)
        U = rng.random()
        true_bin = 0
        for i in range(len(cum_dist)):
            if U <= cum_dist[i]:
                true_bin = int(lung_classes[i])
                break
        if true_bin <= 1:
            dose = 0.0;
        elif true_bin >= len(dbins):
            dose = dbins[-1]
        else:
            bin_low = dbins[true_bin-1]
            bin_high = dbins[true_bin]
            U = rng.random()
            dose = (bin_high - bin_low)*U + bin_low
        #
        y_next_lung_bin.append(true_bin)
        y_next_lung.append(dose)
    df_person_start["lung_next_guess"] = y_next_lung
    df_person_start["lung_bin_next"] = y_next_lung_bin
    # ------------------------------------------------------------------ # Esophagus
    X_next_eso = df_person_start[x_col_eso]

    y_next_eso_dist = base_rf_eso.predict_proba(X_next_eso)

    y_next_eso_bin = []
    y_next_eso = []
    for dist in y_next_eso_dist:
        cum_dist = np.cumsum(dist)
        U = rng.random()
        true_bin = 0
        for i in range(len(cum_dist)):
            if U <= cum_dist[i]:
                true_bin = int(eso_classes[i])
                break
        if true_bin <= 1:
            dose = 0.0;
        elif true_bin >= len(dbins):
            dose = dbins[-1]
        else:
            bin_low = dbins[true_bin-1]
            bin_high = dbins[true_bin]
            U = rng.random()
            dose = (bin_high - bin_low)*U + bin_low
        #
        y_next_eso_bin.append(true_bin)
        y_next_eso.append(dose)
    df_person_start["esophagus_next_guess"] = y_next_eso
    df_person_start["esophagus_bin_next"] = y_next_eso_bin
    # ---------------------------------------------------------- #
    df_person_start.to_csv("ACCURACY/Subj_RF_"+str(subj_i)+".csv")
