# ExtendedPlasmode
Houses code for a paper on exposure generating-functions for plasmode simulations of radiation cohorts.

## SYNTHETIC_DATA.py

Python script which generates synthetic data, mimics the structure of the original data. Note that it does not replicate the original data and is intended to show the columns in the data.

Creates seven csv files for the person data and six organ dose data sets.

## Generate_Extended_Data.py

Python script that fully generates the extended data. Loads the synthetic data (creating data for risk regression), creates the data used for estiamting annual doses, and applies the extension models. Note that the parametric method is hard coded, and will extend observations using the original models. The random forest models are trained during the run, so they will predict annual doses using the synthetic data. Code is provided for both the extension (predicting out to age 95) and the next year predictions.

Creates several csv files in the main folder for regressions on dose (df\_total.csv) and storing the calculated annual doses (preped\_data.csv). Extensions are written to the NO_EVENTS folder, accuracy predictions are written to the ACCURACY folder.

## run_reg.R

R script that calculates the dose response and background cumulative hazard models. Uses the Colossus package to fit each regression, writes the result table to a text file in the NEW\_RUNS folder, and then saves a table of the cumulative hazards to the NEW\_RUNS folder.

## Assign_Events.py

pyspark script that calculates event times for each plasmode simulation. The event rates are hard coded, so they will match the original data event rates. The data is loaded from the NO\_EVENTS folder and new data is saved to the WITH\_EVENTS folder.

## gen_analyze_scripts.py

python script that writes r scripts to perform regressions on the plasmode simulations. Writes r scripts to the ANALYSIS\_SCRIPTS folder and a bash script to the main folder to automate running everything. The bash script commands save the regression outputs to text files in the RESULTS folder.
