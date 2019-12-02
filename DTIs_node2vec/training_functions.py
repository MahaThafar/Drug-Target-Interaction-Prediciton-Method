'''
*******************************************************
Some functions that are needed for training process..
*******************************************************
'''
from copy import deepcopy
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import scipy.spatial
from sklearn.preprocessing import MinMaxScaler

#---------------------------------------------------------------------------------------------
#### ALL needed functions for training process
#---------------------------------------------------------------------------------------------

def Mask_test_index(test_idx, Ft, Lab,  DrTr, drugID, targetID):
    
    DrTr_train = deepcopy(DrTr)
    # get the drug index and target index 
    # mask drug,target = 1 of test data to be 0 (i.e. remove the edge)
    for i in test_idx:
        if(Lab[i]==1):
            dr = Ft[i,0]
            dr_index = drugID[dr]
            
            tr = Ft[i,1]
            tr_index = targetID[tr]
            
            DrTr_train[tr_index, dr_index] = 0
    
    return DrTr_train
#---------------------------------------------------------------------------------------------

## normalize simiarities to be in positive range [0,1]

def normalizedMatrix(matrix):
    
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(matrix)
    normMat = scaler.transform(matrix)

    return normMat
#---------------------------------------------------------------------------------------------

## To compute drug-drug FV cosine similarity and target-target FV cosine similarity

def Cosine_Similarity(Matrix):
    cos_sim_m = np.zeros((Matrix.shape[0],Matrix.shape[0]))
    for item_i in range(Matrix.shape[0]):
        for item_j in range(Matrix.shape[0]):
            cos_sim_m[item_i][item_j] = 1-(scipy.spatial.distance.cosine(Matrix[item_i,:],Matrix[item_j,:]))
    
    return cos_sim_m
#---------------------------------------------------------------------

## To find drug-target cosine similarities from their FV and add it to be one feature for the classifier

def DT_Cosine_Sim(M1,M2):
    cos_sim_m = np.zeros((M1.shape[0],M2.shape[0]))
    for item_i in range(M1.shape[0]):
        for item_j in range(M2.shape[0]):
            cos_sim_m[item_i][item_j] = 1-(scipy.spatial.distance.cosine(M1[item_i,:],M2[item_j,:]))
    
    return cos_sim_m
#---------------------------------------------------------------------

## Convert similarity matrix to dataframe edge list

def edgeList(simMat, items):
    EL = []
    for i in range(0,len(simMat)):
        for j in range(0,len(simMat)):
            item1 = items[i]
            item2 = items[j]
            sim = simMat[i][j]
            pairSim = item1, item2, sim
            EL.append(pairSim)
        
    df = pd.DataFrame(EL)

    # remove al edges with similarity =0
    sim_df = df[(df.T != 0.0).all()]

    return sim_df
#----------------------------------------

def Strongest_k_sim(Mat,K):
    
    m,n = Mat.shape

    Ssim = np.zeros((m,n))
    for i in range(m):
        index =  np.argsort(Mat[i,:])[-K:] # sort based on strongest k edges
        Ssim[i,index] = Mat[i,index] # keep only the nearest neighbors (strongest k edges)
    
    np.fill_diagonal(Ssim , 1)  
     
    return Ssim
#----------------------------------------------