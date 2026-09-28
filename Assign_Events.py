import pandas as pd
import numpy as np
import pyspark.pandas as ps
from pyspark.sql import SparkSession, Window, SQLContext
from pyspark.sql import functions as F

from pyspark import SparkConf, SparkContext

from pyspark.sql.functions import udf, struct, when, rand, randn, exp, log, pow, lit, monotonically_increasing_id
from pyspark.sql.types import IntegerType, NumericType

import os
import glob
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# below is the console command I use. It sets the number of cores to all available and tells it that it can use 22g of ram
# spark-submit --master local[*] --driver-memory 20g --executor-memory 2g --verbose Assign_Events.py

import time
#
conf = SparkConf().setAppName("AppBase")
sc=SparkContext(conf=conf)
sc.setLogLevel("WARN")
sqlContext = SQLContext(sc)
#

def Write_EXP_Event(iname, oname, doses, paras_events, paras_censor,seed=3742621):
#    tstart = time.time()
    psdf0 = sqlContext.read.option("header", True).csv(iname)
    windowval = (Window.partitionBy('StudyID').orderBy('age_entry')
                 .rowsBetween(Window.unboundedPreceding, 0))
    # Then the total dose is just the sum
    # Next I just define all of the factor covariates, you can change them as needed.
    psdf0 = psdf0.withColumn("sex0", when(psdf0.sexm==0,1).otherwise(0))
    psdf0 = psdf0.withColumn("sex1", when(psdf0.sexm==1,1).otherwise(0))
    #
    psdf0 = psdf0.withColumn("SES_Low", when(psdf0.SES_HighestEd=="Low",1).otherwise(0))
    psdf0 = psdf0.withColumn("SES_Medium", when(psdf0.SES_HighestEd=="Medium",1).otherwise(0))
    psdf0 = psdf0.withColumn("SES_High", when(psdf0.SES_HighestEd=="High",1).otherwise(0))
    #
    psdf0 = psdf0.withColumn("yob_cat_0", when(psdf0.yob_cat10==0,1).otherwise(0))
    psdf0 = psdf0.withColumn("yob_cat_1", when(psdf0.yob_cat10==1,1).otherwise(0))
    psdf0 = psdf0.withColumn("yob_cat_2", when(psdf0.yob_cat10==2,1).otherwise(0))
    psdf0 = psdf0.withColumn("yob_cat_3", when(psdf0.yob_cat10==3,1).otherwise(0))
    psdf0 = psdf0.withColumn("yob_cat_4", when(psdf0.yob_cat10==4,1).otherwise(0))
    psdf0 = psdf0.withColumn("yob_cat_5", when(psdf0.yob_cat10==5,1).otherwise(0))
    psdf0 = psdf0.withColumn("yob_cat_6", when(psdf0.yob_cat10==6,1).otherwise(0))
    psdf0 = psdf0.withColumn("yob_cat_7", when(psdf0.yob_cat10==7,1).otherwise(0))
    psdf0 = psdf0.withColumn("yob_cat_8", when(psdf0.yob_cat10==8,1).otherwise(0))
    psdf0 = psdf0.withColumn("yob_cat_9", when(psdf0.yob_cat10==9,1).otherwise(0))
    # ----------------- Allocate the event and exit_adjusts ---------------- # MAIN EVENTS
    run_num = 0
    select_col = ['StudyID', 'age_entry','age_exit','sexm','yob_cat10', 'SES_HighestEd']
    for col in ['Esophagus', 'lung', 'Parkinsons', 'Ischemic', 'CLL', 'AML', 'NonCLL_excAML', "Heart_excIsch", "AllLymphHema_excLeukAleuk", "AllCancer_excElse"]:
        select_col.append(doses[col])
        run_num += 1
        name0 = "event_"+col
        name1 = "exit_adjust_" + col
        name2 = "p_event_" + col
        select_col.append(name0)
        select_col.append(name1)
#        select_col.append(name2)
        psdf0=psdf0.withColumn(name0, lit(0))
        psdf0=psdf0.withColumn(name1, lit(0.0))
        # ------------------------------------------------------------------------------------------------------------------------------- #
        para_list = paras_events[col]
        para_list = [float(i) for i in para_list]
        psdf0=psdf0.withColumn("back_rate0", para_list[0]*pow(psdf0.age_entry,2)/1000+para_list[1]*psdf0.age_entry + para_list[2])
        psdf0=psdf0.withColumn("back_rate0", exp(psdf0.back_rate0))
        psdf0=psdf0.withColumn("back_rate1", para_list[0]*pow(psdf0.age_exit,2)/1000+para_list[1]*psdf0.age_exit + para_list[2])
        psdf0=psdf0.withColumn("back_rate1", exp(psdf0.back_rate1))
        psdf0=psdf0.withColumn("back_rate", psdf0.back_rate1 - psdf0.back_rate0)
        # -------------------------------------------------------------------------------------------------------------------------------- #
        cov_cols = ["yob_cat_1", "yob_cat_2", "yob_cat_3", "yob_cat_4", "yob_cat_5", "yob_cat_6", "yob_cat_7", "yob_cat_8", "yob_cat_9",
                    "SES_Low", "SES_Medium", "sexm"] # here are the columns it expects to use for non-dose
        psdf0=psdf0.withColumn("cov_rate", lit(0.0)) # The intitial covariates are log-linear
        for i in range(len(cov_cols)):
            psdf0=psdf0.withColumn("cov_rate", psdf0.cov_rate + psdf0[cov_cols[i]]*para_list[i+3])
        psdf0=psdf0.withColumn("cov_rate", (lit(1) + para_list[-1] * psdf0[doses[col]]) * exp(psdf0.cov_rate))
        # ------------------------------------------------------------------------------------------------------------------------------ #
        psdf0=psdf0.withColumn(name2, psdf0.cov_rate * psdf0.back_rate)
        psdf0=psdf0.withColumn("unirand", rand(seed + 79*run_num))
        psdf0=psdf0.withColumn(name1, psdf0.age_entry-log(psdf0.unirand)/psdf0[name2])
        psdf0=psdf0.withColumn(name0, when(psdf0.age_exit>psdf0[name1],1).otherwise(0))
        psdf0=psdf0.withColumn(name1, when(psdf0[name0]>0,psdf0[name1]).otherwise(psdf0.age_exit))
    # ----------------- Allocate the event and exit_adjusts ---------------- # censoring
    for group in [0,1,2,3,4,5,6,7,8,9]:
        name0 = "rate_" + str(group)
        para_list = paras_censor[group]
        para_list = [float(i) for i in para_list]
        # ------------------------ # YOB 0
        psdf0=psdf0.withColumn("back_rate0", when(psdf0.age_entry < para_list[3], para_list[0]*pow(psdf0.age_entry,2)/1000+para_list[1]*psdf0.age_entry + para_list[2]).otherwise(
                                            para_list[0]*pow(para_list[3],2)/1000+para_list[1]*para_list[3] + para_list[2] + para_list[4] * (psdf0.age_entry - para_list[3]))
                              )
        psdf0=psdf0.withColumn("back_rate0", exp(psdf0.back_rate0))
        psdf0=psdf0.withColumn("back_rate1",  when(psdf0.age_exit < para_list[3], para_list[0]*pow(psdf0.age_exit,2)/1000+para_list[1]*psdf0.age_exit + para_list[2]).otherwise(
                                            para_list[0]*pow(para_list[3],2)/1000+para_list[1]*para_list[3] + para_list[2] + para_list[4] * (psdf0.age_exit - para_list[3]))
                              )
        psdf0=psdf0.withColumn("back_rate1", exp(psdf0.back_rate1))
        psdf0=psdf0.withColumn("back_rate", psdf0.back_rate1 - psdf0.back_rate0)
        # -------------------------------------------------------------------------------------------------------------------------------- #
        cov_cols = ["SES_Low", "SES_Medium", "sexm"] # here are the columns it expects to use for non-dose
        psdf0=psdf0.withColumn("cov_rate", lit(0.0)) # The intitial covariates are log-linear
        for i in range(len(cov_cols)):
            psdf0=psdf0.withColumn("cov_rate", psdf0.cov_rate + psdf0[cov_cols[i]]*para_list[i+5])
        psdf0=psdf0.withColumn("cov_rate", exp(psdf0.cov_rate))
        # ------------------------------------------------------------------------------------------------------------------------------ #
        psdf0=psdf0.withColumn(name0, psdf0.cov_rate * psdf0.back_rate)
    # ------------------------ # Select correct background
    run_num += 1
    name0 = "event_censor"
    name1 = "exit_adjust_censor"
    name2 = "p_event_censor"
    select_col.append(name0)
    select_col.append(name1)
#    select_col.append(name2)
    psdf0=psdf0.withColumn("p_event_censor", when(psdf0.yob_cat10==0, psdf0.rate_0).otherwise(
                                            when(psdf0.yob_cat10==1, psdf0.rate_1).otherwise(
                                            when(psdf0.yob_cat10==2, psdf0.rate_2).otherwise(
                                            when(psdf0.yob_cat10==3, psdf0.rate_3).otherwise(
                                            when(psdf0.yob_cat10==4, psdf0.rate_4).otherwise(
                                            when(psdf0.yob_cat10==5, psdf0.rate_5).otherwise(
                                            when(psdf0.yob_cat10==6, psdf0.rate_6).otherwise(
                                            when(psdf0.yob_cat10==7, psdf0.rate_7).otherwise(
                                            when(psdf0.yob_cat10==8, psdf0.rate_8).otherwise(
                                            psdf0.rate_9)) ))))))))
    psdf0=psdf0.withColumn("unirand", rand(seed + 79*run_num))
    psdf0=psdf0.withColumn("exit_adjust_censor", psdf0.age_entry-log(psdf0.unirand)/psdf0.p_event_censor)
    psdf0=psdf0.withColumn("event_censor", when(psdf0.age_exit>psdf0.exit_adjust_censor,1).otherwise(0))
    psdf0=psdf0.withColumn("exit_adjust_censor", when(psdf0.event_censor>0,psdf0.exit_adjust_censor).otherwise(psdf0.age_exit))
    # ------------------------------------------------------------------------------------------------------------------------------- # return
    select_col = list(np.unique(select_col))
    psdf0.select(select_col).write.option("header",True).parquet(oname, 'overwrite')
    return



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
#                    columns  |----------- background ------| ['yob_1',   'yob_2',    'yob_3',    'yob_4',  'yob_5',   'yob_6',   'yob_7',   'yob_8',   'yob_9',  'SES_low', 'SES_medium, sexm,     dose   ]
paras_events = {'Esophagus':[-1.77E+00,  3.47E-01, -2.21E+01,  5.95E-01,  8.78E-01,  9.25E-01,  8.37E-01,  5.28E-01,  7.81E-01,  6.08E-01,  5.51E-01,  6.00E-01,  4.78E-01,  2.66E-01, -5.98E-01,  2.16E+00], 
                     'lung':[-1.43E+00,  2.96E-01, -1.74E+01,  4.40E-02,  3.57E-02,  7.30E-02, -5.34E-02, -1.31E-01, -3.02E-01, -5.25E-01, -5.13E-01, -9.03E-01,  7.92E-01,  5.77E-01, -2.82E-01, -5.31E-01], 
               'Parkinsons':[-5.90E+00,  1.17E+00, -6.09E+01, -2.88E-01, -1.00E-01,  1.70E-01,  1.58E-01,  2.01E-01,  2.23E-01, -2.85E-01,  0.00e+00,  0.00e+00, -3.95E-01, -1.43E-01, -1.25E+00,  2.30E+00], 
                 'Ischemic':[-6.02E-02,  1.02E-01, -9.79E+00,  1.60E-02,  1.42E-02, -8.91E-02, -2.52E-01, -3.78E-01, -4.72E-01, -5.32E-01, -4.85E-01, -4.82E-01,  5.78E-01,  4.12E-01, -9.06E-01, -1.06E-01], 
                      'CLL':[ 0.00E+00,  1.75E-01, -2.21E+01,  8.91E-01,  1.29E+00,  1.98E+00,  2.12E+00,  1.55E+00,  1.37E+00,  1.44E+00,  8.33E-01,  0.00e+00,  3.97E-01,  2.12E-01,  0.00e+00,  1.00E-01], 
                      'AML':[ 0.00E+00,  7.82E-02, -1.24E+01,  5.31E-01,  1.87E-01,  2.28E-01,  3.28E-01,  7.67E-02,  4.80E-01,  2.68E-01,  4.03E-01, -4.61E-01,  3.18E-01,  1.55E-01, -4.62E-01,  3.64E+00], 
            'NonCLL_excAML':[ 0.00E-01,  9.01E-02, -1.31E+01,  3.90E-01,  1.01E-01,  4.13E-01,  8.58E-02,  1.42E-01,  1.98E-01, -1.81E-03, -9.68E-02, -5.33E-01,  1.94E-01, -8.74E-02,  2.49E-01,  1.33E-01], 
            'Heart_excIsch':[ 8.16E-02,  9.00E-02, -1.04E+01, -3.27E-03, -8.68E-03, -7.53E-02, -5.39E-02, -9.78E-02, -8.72E-02, -1.12E-01,  5.42E-02, -8.88E-02,  6.36E-01,  4.50E-01, -4.17E-01, -4.79E-01], 
'AllLymphHema_excLeukAleuk':[-1.23E-01,  1.09E-01, -1.25E+01, -4.43E-02,  1.74E-01,  2.85E-02,  1.92E-01,  7.26E-02,  1.08E-01, -3.39E-02,  1.92E-01, -7.34E-01,  2.40E-01,  1.43E-01, -2.87E-01, -2.59E-01], 
        'AllCancer_excElse':[-8.12E-01,  2.13E-01, -1.42E+01,  1.17E-01,  1.22E-01,  1.93E-02, -6.25E-02, -6.59E-02, -9.12E-02, -1.35E-01, -1.86E-01, -1.61E-01,  3.73E-01,  2.66E-01,  5.37E-02,  4.30E-01]
}
#        columns  |----------- background -----------------------------------------------|  ['SES_low','SES_Medium','sexm'   ]
paras_censor = {0:[-0.9118052368077578,  0.24610932503697477, -15.600364041711819, 90, 0.08, 0.02433259, 0.03190880, 0.08003213],
                1:[-0.14254538904775949, 0.13561099609327454, -11.661311412538623, 90, 0.4,  0.02433259, 0.03190880, 0.08003213],
                2:[-0.2753545962172861,  0.15705464545962455, -12.535428820608196, 85, 0.5,  0.02433259, 0.03190880, 0.08003213],
                3:[-0.12291842720052683, 0.11104497663534708,  -9.764650257255694, 80, 0.55, 0.02433259, 0.03190880, 0.08003213],
                4:[-0.584485922953571,   0.1733252479011444,  -11.882247152057436, 75, 0.59, 0.02433259, 0.03190880, 0.08003213],
                5:[-0.40611517363181343, 0.12965767831735195,  -9.7252345344795,   70, 0.9,  0.02433259, 0.03190880, 0.08003213],
                6:[-0.49150500849456846, 0.11883929702600396,  -8.558369718041083, 65, 0.9,  0.02433259, 0.03190880, 0.08003213],
                7:[-1.035859903340698,   0.16290973665439493,  -9.171534056542523, 60, 0.9,  0.02433259, 0.03190880, 0.08003213],
                8:[-1.16443269231425,    0.1676315625417529,   -8.992212360123283, 55, 0.9,  0.02433259, 0.03190880, 0.08003213],
                9:[-1.3112881696966627,  0.17453565757171569,  -8.889464845047193, 47, 0.52, 0.02433259, 0.03190880, 0.08003213]
}

#
#
for i in range(500):
    for j_name in ["CONST","PARA","RF"]:
        print("Event: " + str(i) + " " + j_name)
        iname = "NO_EVENTS/Subj_"+str(i)+"_"+j_name+".csv"
        oname = "WITH_EVENTS/Subj_"+str(i)+"_"+j_name+""
        if os.path.exists(oname):
            print("exists")
        else:
            pass
            Write_EXP_Event(iname, oname, doses, paras_events, paras_censor, 3742621 + 30*i)
        
'''
Parkinsons yob_8 and yob_9 changed to 0 due to high uncertainties
CLL yob_9 and sexm changed to 0 due to high uncertainties
''' 
        
        
        
