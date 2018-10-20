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

feature extracted per stroke: 1) each substroke is broken into equidistant segments, the end points are recorded
			      2) we work with two segments at each point of time. [si-1, si, si+1]
			      3) we length of straight line joining si-1 and si+1 (euclidean distance) = L
			      4) now we find out the perpendicular distance of si from the st line joining si-1 and si+1 = w
			      5) we find ratio r1 = L/w
			      6) we add all ratios ri for all consecutive segments, this is our feature representing the substroke.
			      7) our second feauture shall be L


"""

#TODO : The equal distance calculation must be wrt the entire dataset
#what we will do is find the min of the number of datapoints in each stroke. We will divide all others by this min.

def equal_distance(len):
	if(len < 5):
		return 1
		
	return len/5
	
def sqrt(n):
	return n**(1/2.0)
	
def sqr(n):
	return n**2
	
def euclidean(a, b):
	return sqrt(sqr(a)+sqr(b))
	
def find_features(equidistant, index, csv1, csv2,csv3):
	sum_L =0
	sum_r =0
	ratio = 0
	L = 0
	no_of_zero =0
	for i in range(len(equidistant)):
		arr = equidistant[i]
		L = euclidean(arr[0][1]-arr[0][0], arr[2][1]-arr[2][0])
		#slope = (arr[2][1]-arr[0][1])/(arr[2][0]-arr[0][0])
		mid_x = (arr[2][0]+arr[0][0])/2
		mid_y = (arr[2][1]+arr[0][1])/2
		w = euclidean(arr[1][0]-mid_x, arr[1][1]-mid_y)
				
		if w!=0:
			ratio = L/w
		
		sum_r+=ratio
		sum_L+=L
		
	if sum_r ==0 :
		no_of_zero =1
		
	csv1.append(sum_r)
	csv2.append(sum_L)
	csv3.append(index)
	#return ratio
	return no_of_zero
def find_equidistant_extract_features(h5file):
	f = h5py.File(h5file, 'r')
	keys = f.keys()
	total=len(keys)   #Number of character samples in the file.
	csv1 = []
	csv2 = []
	csv3 = []
	#file_names = []
	no_of_zeros =0
	index =0
	for t in range(total):
		equidistant = []    #List which will store all the equidistant points and their corr adj points also.
		group=f.get(keys[t])	#gives the group name, in which "Stroke resides"
		fe = group.get("Stroke")
		fe = np.array(fe)
		l = len(fe)		#number of points
		
		lapse =0	
				
		e = equal_distance(l)
		#p = 10   # considering only 10 points from entire set, all are equidistant
		
		list_equal = []   #List which will store the point and its adjacent points;
		
		j =e            #We start with second point in the set 
		
		while j+e<l:
			listA = []
			listA += [fe[j-e], fe[j], fe[j+e]]	#equidistant points are end points of each segment
			listA = np.asarray(listA).tolist()
			#listA += fe[j]
			#listA += fe[j+1]
			list_equal += [listA]
			j+=e
			#p-=1	
		
		#print(len(list_equal))
		#print(list_equal[2])
		equidistant.append(list_equal)	#contains for all strokes making the character.	
		#print(equidistant[0][1])
		
		#tangents = compute_angle(equidistant)
		feature_extracted = find_features(equidistant[0], index, csv1, csv2, csv3)
		no_of_zeros+=feature_extracted
		index+=1
		#print(feature_extracted)
		#print(index)
		
	##To store result in csv:
	df = pd.DataFrame(data={"col1": csv1, "col2": csv2, "col3": csv3})
	df.to_csv("./feature_array_curliness.csv", sep=',',index=False)
	print no_of_zeros
	

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

	data = pd.read_csv('feature_array_curliness.csv')
	X = data.iloc[:,:2].values
	x1 = data.iloc[:,:1].values
	x2= data.iloc[:,1:2].values
	
	x1 = x1.tolist()
	x2 = x2.tolist()
	#print(X)
	#label_list =[]
	
	gmm1 = GMM(n_components = n_clusters, init_params = 'kmeans', max_iter = 5000, covariance_type = 'diag').fit(X)
	label_gmm_1 = gmm1.predict(X)
	#####
	"""
	for i in range(len(label_gmm_1)):
		label_list.append([X[i][0],X[i][1], label_gmm_1[i]])
	"""
	#####
		
	df = pd.DataFrame(data={"col1": x1, "col2": x2, "col3": label_gmm_1})
	df.to_csv("./final_curliness.csv", sep=',',index=False)
	#print "lap"
	#print lap
	
#read_HDF("/home/aiswarya/OHRSegmentation/Data/Unlabelled_Segmented_Train.h5")

find_equidistant_extract_features("/home/aiswarya/OHRSegmentation/Data/Unlabelled_Segmented_Train.h5")
#read_HDF("/home/aiswarya/OHRSegmentation/Data/Train_resamp_us.h5")
GMM_results(16)

