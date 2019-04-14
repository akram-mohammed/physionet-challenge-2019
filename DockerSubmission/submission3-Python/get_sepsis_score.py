#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import os, shutil, zipfile
from numpy import array
import csv
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import entropy
import scipy as sc
from zipfile import ZipFile
from sklearn.externals import joblib	

def get_sepsis_score(data1):

	# Load the saved model pickle file
	Trained_model = joblib.load('saved_model.pkl') 
	
	#Testing
	df_test = data1
	#Forward fill missing values
	df_test.fillna(method='ffill', axis=0, inplace=True)
	df_test = pd.DataFrame(df_test).fillna(0)
	#count = 0
	df_test['ID'] = 0
	DBP = pd.pivot_table(df_test,values='DBP',index='ID',columns='ICULOS')
	O2Sat = pd.pivot_table(df_test,values='O2Sat',index='ID',columns='ICULOS')
	Temp = pd.pivot_table(df_test,values='Temp',index='ID',columns='ICULOS')
	RR = pd.pivot_table(df_test,values='Resp',index='ID',columns='ICULOS')
	BP = pd.pivot_table(df_test,values='SBP',index='ID',columns='ICULOS')
	latest = pd.pivot_table(df_test,values='HR',index='ID',columns='ICULOS')
	Heart_rate_test = latest 
	RR_test = RR 
	BP_test = BP 
	DBP_test = DBP 
	Temp_test = Temp 
	O2Sat_test = O2Sat 

	result = Heart_rate_test

	result = result.fillna(0)
	RR_test = RR_test.fillna(0)
	BP_test = BP_test.fillna(0)
	Temp_test = Temp_test.fillna(0)
	DBP_test = DBP_test.fillna(0)
	O2Sat_test = O2Sat_test.fillna(0)
	
	#Since we are using a windows-based approach (6-hour window size), we pad our output for the 6 hours following patients admission.
	scores_list = [0.9,0.9,0.9,0.9,0.9,0.9]
	labels_list = [1,1,1,1,1,1]

	scores1 = []
	labels1 = []
	#Get dataframe of probs
	#Windows based approach
	for iterat in range(0,RR_test.shape[1]-6): 
		
		for i in range (iterat,iterat+1): 
			Heart_rate_test = result.iloc[:, i:i+6]
			RR2_test = RR_test.iloc[:, i:i+6]
			BP2_test = BP_test.iloc[:, i:i+6]
			Temp2_test = Temp_test.iloc[:, i:i+6]
			DBP2_test = DBP_test.iloc[:, i:i+6]
			O2Sat2_test = O2Sat_test.iloc[:, i:i+6]

			result['HR_min'] = Heart_rate_test.min(axis=1)
			result['HR_mean'] = Heart_rate_test.mean(axis=1)
			result['HR_max'] = Heart_rate_test.max(axis=1)
			result['HR_stdev'] = Heart_rate_test.std(axis=1)
			result['HR_var'] = Heart_rate_test.var(axis=1)
			result['HR_skew'] = Heart_rate_test.skew(axis=1)
			result['HR_kurt'] = Heart_rate_test.kurt(axis=1)
			
			result['BP_min'] = BP2_test.min(axis=1)
			result['BP_mean'] = BP2_test.mean(axis=1)
			result['BP_max'] = BP2_test.max(axis=1)
			result['BP_stdev'] = BP2_test.std(axis=1)
			result['BP_var'] = BP2_test.var(axis=1)
			result['BP_skew'] = BP2_test.skew(axis=1)
			result['BP_kurt'] = BP2_test.kurt(axis=1)

			result['RR_min'] = RR2_test.min(axis=1)
			result['RR_mean'] = RR2_test.mean(axis=1)
			result['RR_max'] = RR2_test.max(axis=1)
			result['RR_stdev'] = RR2_test.std(axis=1)
			result['RR_var'] = RR2_test.var(axis=1)
			result['RR_skew'] = RR2_test.skew(axis=1)
			result['RR_kurt'] = RR2_test.kurt(axis=1)

			result['DBP_min'] = DBP2_test.min(axis=1)
			result['DBP_mean'] = DBP2_test.mean(axis=1)
			result['DBP_max'] = DBP2_test.max(axis=1)
			result['DBP_stdev'] = DBP2_test.std(axis=1)
			result['DBP_var'] = DBP2_test.var(axis=1)
			result['DBP_skew'] = DBP2_test.skew(axis=1)
			result['DBP_kurt'] = DBP2_test.kurt(axis=1)

			result['O2Sat_min'] = O2Sat2_test.min(axis=1)
			result['O2Sat_mean'] = O2Sat2_test.mean(axis=1)
			result['O2Sat_max'] = O2Sat2_test.max(axis=1)
			result['O2Sat_stdev'] = O2Sat2_test.std(axis=1)
			result['O2Sat_var'] = O2Sat2_test.var(axis=1)
			result['O2Sat_skew'] = O2Sat2_test.skew(axis=1)
			result['O2Sat_kurt'] = O2Sat2_test.kurt(axis=1)

			result['Temp_min'] = Temp2_test.min(axis=1)
			result['Temp_mean'] = Temp2_test.mean(axis=1)
			result['Temp_max'] = Temp2_test.max(axis=1)
			result['Temp_stdev'] = Temp2_test.std(axis=1)
			result['Temp_var'] = Temp2_test.var(axis=1)
			result['Temp_skew'] = Temp2_test.skew(axis=1)
			result['Temp_kurt'] = Temp2_test.kurt(axis=1)
	 
			X_test = result.values[:, Temp2_test.shape[1]:Temp2_test.shape[1]+42] 

			scores = Trained_model.predict_proba(X_test)
			scores1.append(scores[0][1])
			
			if scores1[0]>=0.3:
				labels = 1
			else:
				labels = 0
			labels1.append(labels)
	return (scores_list+scores1, labels_list+labels1)

def read_challenge_data(input_file):
    data1 = pd.DataFrame()
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]
        data1 = pd.DataFrame(data, columns=column_names)
    return data1

if __name__ == '__main__':
    # read data
    data1 = read_challenge_data(sys.argv[1])

    # make predictions
    if data1.size != 0:
        (scores, labels) = get_sepsis_score(data1)

    # write results
    with open(sys.argv[2], 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        if data1.size != 0:
            for (s, l) in zip(scores, labels):
                f.write('%g|%d\n' % (s, l))
