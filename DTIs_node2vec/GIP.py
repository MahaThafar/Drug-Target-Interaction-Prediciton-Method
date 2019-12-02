'''
************************************************************
GIP functions that are needed to calculate Gussian interaction profile kernel
for each pair of drugs and each pair of targets.

Based on the paper:
van Laarhoven, Twan, Sander B. Nabuurs, and Elena Marchiori.
"Gaussian interaction profile kernels for predicting drug-target interaction"
Bioinformatics 27.21 (2011): 3036-3043. 

***********************************************************
'''
import numpy as np
import itertools
from copy import deepcopy
from math import exp
#----------------------------------------------------

def impute_zeros(inMat,inSim,k=5):
	
	mat = deepcopy(inMat)
	sim = deepcopy(inSim)
	(row,col) = mat.shape
	np.fill_diagonal(mat,0)

	indexZero = np.where(~mat.any(axis=1))[0]
	numIndexZeros = len(indexZero)

	np.fill_diagonal(sim,0)
	if numIndexZeros > 0:
		sim[:,indexZero] = 0
	for i in indexZero:
		currSimForZeros = sim[i,:]
		indexRank = np.argsort(currSimForZeros)

		indexNeig = indexRank[-k:]
		simCurr = currSimForZeros[indexNeig]

		mat_known = mat[indexNeig, :]
		
		if sum(simCurr) >0:  
			mat[i,: ] = np.dot(simCurr ,mat_known)# / sum(simCurr)
	
	return mat
#----------------------------------------------------

def func(x):
	return exp(-1*x)
#----------------------------------------------------

def Get_GIP_profile(adj,t):
	'''It assumes target drug matrix'''
	bw = 1
	if t == "d": #profile for drugs similarity
		ga = np.dot(np.transpose(adj),adj)

	elif t=="t": #profile for target similarity
		ga = np.dot(adj,np.transpose(adj))
		
	else:
		sys.exit("The type is not supported: %s"%t)

	ga = bw*ga/np.mean(np.diag(ga))
	di = np.diag(ga)
	x =  np.tile(di,(1,di.shape[0])).reshape(di.shape[0],di.shape[0])
	#z = np.tile(np.transpose(di),(di.shape[0],1)).reshape(di.shape[0],di.shape[0])

	d =x+np.transpose(x)-2*ga
	
	f = np.vectorize(func)

	return f(d)
#----------------------------------------------------