# import numpy as np
# import pandas as pd
# from scipy.sparse import diags
# from scipy.spatial.distance import cdist
# import scipy.stats
# from sklearn.decomposition import PCA

# def _B0_normalized(W, alpha=1.0):
#     """
#     Normalizes `W` so that subjects are always most similar to themselves

#     Parameters
#     ----------
#     W : (N x N) array_like
#         Similarity array from SNF
#     alpha : (0,1) float, optional
#         Factor to add to diagonal of `W` to increase subject self-affinity.
#         Default: 1.0

#     Returns
#     -------
#     W : (N x N) np.ndarray
#         "Normalized" similiarity array
#     """

#     # add `alpha` to the diagonal and symmetrize `W`
#     W = W + (alpha * np.eye(len(W)))
#     #W = check_symmetric(W, raise_warning=False)

#     return W



# def _find_dominate_set(W, K=5):
#     """
#     Parameters
#     ----------
#     W : (N x N) array_like
#     K : (0, N) int, optional
#         Hyperparameter normalization factor for scaling. Default: 20

#     Returns
#     -------
#     newW : (N x N) np.ndarray
#     """

#     m, n = W.shape
#     IW1 = np.flip(W.argsort(axis=1), axis=1)
#     newW = np.zeros(W.size)
#     I1 = ((W[:, :K] * m) + np.vstack(np.arange(n))).flatten(order='F')
#     newW[I1] = W.flatten(order='F')[I1]
#     newW = newW.reshape(W.shape, order='F')
#     newW = newW / newW.sum(axis=1)[:, np.newaxis]

#     return newW

# def SNF(aff, K=5, t=5, alpha=1.0):
#     """
#     Performs Similarity Network Fusion on `aff` matrices

#     Parameters
#     ----------
#     aff : `m`-list of (N x N) array_like
#         Input similarity arrays. All arrays should be square and of equal size.
#     K : (0, N) int, optional
#         Hyperparameter normalization factor for scaling. Default: 20
#     t : int, optional
#         Number of iterations to perform information swapping. Default: 20
#     alpha : (0,1) float, optional
#         Hyperparameter normalization factor for scaling. Default: 1.0

#     Returns
#     -------
#     W: (N x N) np.ndarray
#         Fused similarity network of input arrays
#      """

#     #aff = [check_symmetric(check_array(a)) for a in aff]
#     #check_consistent_length(*aff)

#     m, n = aff[0].shape
#     newW, aff0 = [0] * len(aff), [0] * len(aff)
#     Wsum = np.zeros((m, n))

#     for i in range(len(aff)):
#         aff[i] = aff[i] / aff[i].sum(axis=1)[:, np.newaxis]
#         #aff[i] = check_symmetric(aff[i], raise_warning=False)
#         newW[i] = _find_dominate_set(aff[i], round(K))
#     Wsum = np.sum(aff, axis=0)

#     for iteration in range(t):
#         for i in range(len(aff)):
#             aff0[i] = newW[i] * (Wsum - aff[i]) * newW[i].T / (len(aff) - 1)
#             aff[i] = _B0_normalized(aff0[i], alpha=alpha)
#         Wsum = np.sum(aff, axis=0)

#     W = Wsum / len(aff)
#     W = W / W.sum(axis=1)[:, np.newaxis]
#     W = (W + W.T + np.eye(n)) / 2

#     return W
#--------------------------------------------------
import numpy as np
import copy 
#--------------------------------------------------

def FindDominantSet(W,K):
    m,n = W.shape
    DS = np.zeros((m,n))
    for i in range(m):
        index =  np.argsort(W[i,:])[-K:] # get the closest K neighbors 
        DS[i,index] = W[i,index] # keep only the nearest neighbors 

    #normalize by sum 
    B = np.sum(DS,axis=1)
    B = B.reshape(len(B),1)
    DS = DS/B
    return DS
#--------------------------------------------------

def normalized(W,alpha):
    m,n = W.shape
    W = W+alpha*np.identity(m)
    return (W+np.transpose(W))/2

#--------------------------------------------------

def SNF(Wall,K,t,alpha=1):
    C = len(Wall)
    m,n = Wall[0].shape

    for i in range(C):
        B = np.sum(Wall[i],axis=1)
        len_b = len(B)
        B = B.reshape(len_b,1)
        Wall[i] = Wall[i]/B
        Wall[i] = (Wall[i]+np.transpose(Wall[i]))/2

    newW = []
    
    for i in range(C):
        newW.append(FindDominantSet(Wall[i],K))       

    Wsum = np.zeros((m,n))
    for i in range(C):
        Wsum += Wall[i]

    for iteration in range(t):
        Wall0 = []
        for i in range(C):
            temp = np.dot(np.dot(newW[i], (Wsum - Wall[i])),np.transpose(newW[i]))/(C-1)
            Wall0.append(temp)

        for i in range(C):
            Wall[i] = normalized(Wall0[i],alpha)

        Wsum = np.zeros((m,n))
        for i in range (C):
            Wsum+=Wall[i]

    W = Wsum/C
    B = np.sum(W,axis=1)
    B = B.reshape(len(B),1)
    W/=B
    W = (W+np.transpose(W)+np.identity(m))/2
    return W

#--------------------------------------------------