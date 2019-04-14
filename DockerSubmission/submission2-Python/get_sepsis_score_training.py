#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import os, shutil, zipfile

from zipfile import ZipFile 

def get_sepsis_score(data1):

	import pandas as pd
	import numpy as np
	from numpy  import array
	import csv
	from pandas import DataFrame
	import os
	from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
	from scipy.stats import entropy
	import scipy as sc
	from zipfile import ZipFile

	# specifying the zip file name 
	file_name = "training_setA.zip"
	  
	# opening the zip file in READ mode 
	with ZipFile(file_name, 'r') as zip: 
		# extracting all the files 
		#print('Extracting all the files now...') 
		zip.extractall() 
		#print('Done!')
		
	# specifying the zip file name 
	file_name = "training_setB.zip"
	  
	# opening the zip file in READ mode 
	with ZipFile(file_name, 'r') as zip: 
		# extracting all the files 
		#print('Extracting all the files now...') 
		zip.extractall() 
		#print('Done!') 

	import glob

	path = r'training' # 
	all_files = glob.glob(path + "/*.psv")

	li = []
	count = 0
	for filename in all_files:
		df = pd.read_csv(filename, index_col=None, sep = "|", header=0)
		df['ID']= count
		li.append(df)
		count = count + 1

	frame1 = pd.concat(li, axis=0, ignore_index=True)

	
	path = r'training_setB' # 
	all_files = glob.glob(path + "/*.psv")

	li = []
	count = 0
	for filename in all_files:
		df = pd.read_csv(filename, index_col=None, sep = "|", header=0)
		df['ID']= count
		li.append(df)
		count = count + 1

	frame2 = pd.concat(li, axis=0, ignore_index=True)


	# Just to make sure to include the unique patient IDs
	frame2['ID'] = frame2['ID'] + 21000


	df = pd.concat([frame1, frame2], axis=0)


	counts_list = df['ID'].value_counts().sort_index().values

	all_1 = df.loc[df['SepsisLabel'] == 1]


	df_sepsis_IDs =  df[df.SepsisLabel == 1]


	sepsis_IDs = df_sepsis_IDs.ID.unique()


	sepsis_df = df.ID.isin(sepsis_IDs)
	sepsis_df = df[sepsis_df]


	#Select all non-sepsis patients
	negative_cases = df[~df.ID.isin(sepsis_IDs)]

	#negative_cases.groupby('ID').count().hist(column='ICULOS', bins=[0, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,140,150,160])


	Combined = pd.concat([sepsis_df, negative_cases.iloc[0:107882,:]])  


	#Forward fill missing values
	Combined.fillna(method='ffill', axis=0, inplace=True)
	Combined = pd.DataFrame(Combined).fillna(0)

	DBP = pd.pivot_table(Combined,values='DBP',index='ID',columns='ICULOS')

	np.random.seed(5)  
	from sklearn.utils import shuffle
	DBP = shuffle(DBP)


	O2Sat = pd.pivot_table(Combined,values='O2Sat',index='ID',columns='ICULOS')


	np.random.seed(5)
	from sklearn.utils import shuffle
	O2Sat = shuffle(O2Sat)


	Temp = pd.pivot_table(Combined,values='Temp',index='ID',columns='ICULOS')


	np.random.seed(5)
	from sklearn.utils import shuffle
	Temp = shuffle(Temp)


	RR = pd.pivot_table(Combined,values='Resp',index='ID',columns='ICULOS')


	np.random.seed(5)
	from sklearn.utils import shuffle
	RR = shuffle(RR)

	BP = pd.pivot_table(Combined,values='SBP',index='ID',columns='ICULOS')


	np.random.seed(5)
	from sklearn.utils import shuffle
	BP = shuffle(BP)


	latest = pd.pivot_table(Combined,values='HR',index='ID',columns='ICULOS')


	np.random.seed(5)
	from sklearn.utils import shuffle
	latest = shuffle(latest)


	Heart_rate = latest 
	RR2 = RR 
	BP2 = BP 
	DBP2 = DBP 
	Temp2 = Temp 
	O2Sat2 = O2Sat 


	class_0_1 = list()
	count=0
	while count < latest.shape[0]: #671
		x = any(sepsis_IDs == latest.index[count])
		if x == True:
			class_0_1.append(1)
				
		else:
			class_0_1.append(0)        
		
		count = count + 1

	se2 = pd.Series(class_0_1)

	def ent(data):
		p_data= data.value_counts()/len(data)  
		entropy=sc.stats.entropy(p_data)   
		return entropy

	Heart_rate = latest 
	RR2 = RR 
	BP2 = BP 
	DBP2 = DBP 
	Temp2 = Temp 
	O2Sat2 = O2Sat
		
	latest['HR_min'] = Heart_rate.min(axis=1)
	latest['HR_mean'] = Heart_rate.mean(axis=1)
	latest['HR_max'] = Heart_rate.max(axis=1)
	latest['HR_stdev'] = Heart_rate.std(axis=1)
	latest['HR_var'] = Heart_rate.var(axis=1)
	latest['HR_skew'] = Heart_rate.skew(axis=1)
	latest['HR_kurt'] = Heart_rate.kurt(axis=1)

	latest['BP_min'] = BP2.min(axis=1)
	latest['BP_mean'] = BP2.mean(axis=1)
	latest['BP_max'] = BP2.max(axis=1)
	latest['BP_stdev'] = BP2.std(axis=1)
	latest['BP_var'] = BP2.var(axis=1)
	latest['BP_skew'] = BP2.skew(axis=1)
	latest['BP_kurt'] = BP2.kurt(axis=1)

	latest['RR_min'] = RR2.min(axis=1)
	latest['RR_mean'] = RR2.mean(axis=1)
	latest['RR_max'] = RR2.max(axis=1)
	latest['RR_stdev'] = RR2.std(axis=1)
	latest['RR_var'] = RR2.var(axis=1)
	latest['RR_skew'] = RR2.skew(axis=1)
	latest['RR_kurt'] = RR2.kurt(axis=1)
		
	latest['DBP_min'] = DBP2.min(axis=1)
	latest['DBP_mean'] = DBP2.mean(axis=1)
	latest['DBP_max'] = DBP2.max(axis=1)
	latest['DBP_stdev'] = DBP2.std(axis=1)
	latest['DBP_var'] = DBP2.var(axis=1)
	latest['DBP_skew'] = DBP2.skew(axis=1)
	latest['DBP_kurt'] = DBP2.kurt(axis=1)

	latest['O2Sat_min'] = O2Sat2.min(axis=1)
	latest['O2Sat_mean'] = O2Sat2.mean(axis=1)
	latest['O2Sat_max'] = O2Sat2.max(axis=1)
	latest['O2Sat_stdev'] = O2Sat2.std(axis=1)
	latest['O2Sat_var'] = O2Sat2.var(axis=1)
	latest['O2Sat_skew'] = O2Sat2.skew(axis=1)
	latest['O2Sat_kurt'] = O2Sat2.kurt(axis=1)
		
	latest['Temp_min'] = Temp2.min(axis=1)
	latest['Temp_mean'] = Temp2.mean(axis=1)
	latest['Temp_max'] = Temp2.max(axis=1)
	latest['Temp_stdev'] = Temp2.std(axis=1)
	latest['Temp_var'] = Temp2.var(axis=1)
	latest['Temp_skew'] = Temp2.skew(axis=1)
	latest['Temp_kurt'] = Temp2.kurt(axis=1)

	X = latest.values[:, max(Combined.ICULOS.values):378]   
	Y = class_0_1[:] 

	num_trees = 700
	model = RandomForestClassifier(n_estimators=num_trees, oob_score=True)
	Trained_model = model.fit(X, Y)

	from sklearn.externals import joblib
	# Output a pickle file for the trained model
	joblib.dump(model, 'saved_model.pkl')

	#Testing
	df_test = data1
	#Forward fill missing values
	df_test.fillna(method='ffill', axis=0, inplace=True)
	df_test = pd.DataFrame(df_test).fillna(0)
	print(df_test.columns)
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
    if len(sys.argv) != 2:
        sys.exit('Usage: %s input[.psv]' % sys.argv[0])

    record_name = sys.argv[1]
    if record_name.endswith('.psv'):
        record_name = record_name[:-4]

    # read input data
    input_file = record_name + '.psv'
    data1 = read_challenge_data(input_file)

    # generate predictions
    (scores, labels) = get_sepsis_score(data1)

    # write predictions to output file
    output_file = record_name + '.out'
    with open(output_file, 'w') as f:
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))
