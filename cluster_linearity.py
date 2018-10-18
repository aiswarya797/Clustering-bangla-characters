import numpy as np
import h5py
import pandas as pd
import csv
from sklearn.mixture import GaussianMixture as GMM
#import pandas as pd
#from PIL import Image,ImageDraw
#import cv2

## Basic Module : Render the strokes given, in the HDF File.
lap =0
"""
TODO :

Train_resamp_us.h5 is the dataset. Each data is a word in bangla. Once we run generate_substroke_for_cluster.py, we get Unlabelled_Segmented_Train.h5 file. Here each sample is a part of a stroke from previous file. 

Example : The first sample of the new file is Aditya13_AmdAni_S0_13, the main word is AmdAni, it has multiple strokes S0, S1, S2 etc.
The new file is named so, because now it means, S0 stroke of AmdAni is cut at 13th point position. This is the only one with S0, means S0 stroke is cut into two, at 13th point.

Next, if we see Aditya13_AmdAni_S2_33, and Aditya13_AmdAni_S2_50 are the ones with S2. Means S2 are cut as : 0-33, 33-50 and 50-end.



"""

def equal_distance(len):
	if(len < 10):
		return len
		
	return len/10
	
def sqrt(n):
	return n**(1/2.0)
	
def find_equidistant(h5file):
	f = h5py.File(h5file, 'r')
	keys = f.keys()
	total=len(keys)   #Number of character samples in the file.
	csv1 = []
	csv2 = []
	csv3 = []
	#file_names = []
	equidistant = []    #List which will store all the equidistant points and their corr adj points also.
	index =0
	for t in range(total):
		group=f.get(keys[t])	#gives the group name, in which "Stroke resides"
		fe = group.get("Stroke")
		fe = np.array(fe)
		l = len(fe)
		
		lapse =0	
				
		e = equal_distance(l)
		p = 10   # considering only 10 points from entire set, all are equidistant
		
		list_equal = []   #List which will store the point and its adjacent points;
		
		j =1            #As we are using j-1 in computation;
		
		while j+1<l and p>0:
			listA = []
			listA += [fe[j-1], fe[j], fe[j+1]]
			listA = np.asarray(listA).tolist()
			#listA += fe[j]
			#listA += fe[j+1]
			list_equal += [listA]
			j+=e
			p-=1	
		
		#print(len(list_equal))
		#print(list_equal[2])
		equidistant.append(list_equal)	#contains for all strokes making the character.	
		#print(equidistant)
		
		#tangents = compute_angle(equidistant)
		tangents = find_features(equidistant, index, csv1, csv2, csv3)
		index+=1
		#print(index)
		
	##To store result in csv:
	df = pd.DataFrame(data={"col1": csv1, "col2": csv2, "col3": csv3})
	df.to_csv("./feature_array.csv", sep=',',index=False)
	

"""	
def read_HDF(h5file):
	f = h5py.File(h5file, 'r')
	keys=f.keys()
	print keys
	total=len(keys)
	print("Total ",total," keys")
	for t in range(1):
		group=f.get(keys[t])
		fe = group.get("Stroke")
		print np.array(fe)
		print np.array(len(fe))
		listA =[]
		
		listA.append(np.asarray(fe[1]).tolist())
		listA.append(np.asarray(fe[2]).tolist())
		print(listA)"""
		
def GMM_results(n_clusters):     #Applies GMM library to the data, and prints the labels.

	data = pd.read_csv('feature_array.csv')
	X = data.iloc[:,:2].values
	x1 = data.iloc[:,:1].values
	x2= data.iloc[:,1:2].values
	
	x1 = x1.tolist()
	x2 = x2.tolist()
	#print(X)
	#label_list =[]
	
	gmm1 = GMM(n_components = n_clusters, init_params = 'kmeans', max_iter = 5000, covariance_type = 'diag').fit(X)
	label_gmm_1 = gmm1.predict(X)
	
	"""for i in range(len(label_gmm_1)):
		label_list.append([X[i][0],X[i][1], label_gmm_1[i]])"""
		
	df = pd.DataFrame(data={"col1": x1, "col2": x2, "col3": label_gmm_1})
	df.to_csv("./final.csv", sep=',',index=False)
	print "lap"
	print lap
	
#read_HDF("/home/aiswarya/OHRSegmentation/Data/Unlabelled_Segmented_Train.h5")

find_equidistant("/home/aiswarya/OHRSegmentation/Data/Unlabelled_Segmented_Train.h5")
#read_HDF("/home/aiswarya/OHRSegmentation/Data/Train_resamp_us.h5")
GMM_results(7)

