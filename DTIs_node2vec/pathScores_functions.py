'''
*****************************************************************************************
Find metapath scores for 6 path structures using NUMPY 3D matrix multiplication,,
    6 path structure:
    1- D-D-T
    2- D-D-D-T
    3- D-T-T
    4- D-T-T-T
    5- D-D-T-T
    6- D-T-D-T
the multiplying path similarity or interaction scores obtained by matrix multiplication then
2 types of features are extracted:
sum of path scores (accumulated scores) and max of the path scores using np.sum, np.max.
******************************************************************************************
'''
import numpy as np

######################### MetaPath Functions ################################

# This function is used for path score of length 4 to add DD or TT matrix multiplication
def DDD_TTT_sim(simM):
    
    np.fill_diagonal(simM,0)
    m = np.einsum('ij,jk->ijk', simM, simM)
    
    sumM = np.sum((m[:, :, None] ), axis = 1)
    maxM = np.max((m[:, :, None] ), axis = 1)
    #avgM = np.mean((m[:, :, None] ), axis = 1)
    
    sumM = np.squeeze(sumM)
    maxM = np.squeeze(maxM)
    #avgM = np.squeeze(avgM)
               
    return (sumM,maxM)#,avgM)
#----------------------------------------------------------

# drugs similarity matrix * training DTIs matrix
def metaPath_Dsim_DT(Dsim,DT,length, mul=False):
    
    np.fill_diagonal(Dsim,0)
    m = np.einsum('ij,jk->ijk', Dsim, DT)

    if(mul):
        m = m**(length)

    sumM = np.sum((m[:, :, None] ), axis = 1)
    maxM = np.max((m[:, :, None]), axis = 1)
    #avgM = np.mean((m[:, :, None] ), axis = 1)

    # to convert from 3-d matrix to 2-d matrix
    sumM = np.squeeze(sumM)
    maxM = np.squeeze(maxM)
    #avgM = np.squeeze(avgM)

    return (sumM,maxM)#,avgM)
#------------------------------------------------------------

#  Training DTIs matrix * targets similarity matrix 
def metaPath_DT_Tsim(Tsim,DT, length, mul=False):

    np.fill_diagonal(Tsim,0)
    
    m = np.einsum('ij,jk->ijk', DT,Tsim)

    if(mul):
        m = m**(length)

    sumM = np.sum((m[:, :, None]), axis = 1)
    maxM = np.max((m[:, :, None] ), axis = 1)
    #avgM = np.mean((m[:, :, None] ), axis = 1)
    

    sumM = np.squeeze(sumM)
    maxM = np.squeeze(maxM)
    #avgM = np.squeeze(avgM) 

    return (sumM,maxM)#, avgM)
#-------------------------------------------------------------

def metaPath_DDTT(DT,Dsim,Tsim, mul=False):

    sumDDT,maxDDT = metaPath_Dsim_DT(Dsim,DT, 3,mul)
    sumDDTT,_ = metaPath_DT_Tsim(Tsim,sumDDT,3,mul)
    _,maxDDTT = metaPath_DT_Tsim(Tsim,maxDDT,3,mul)
    # _,_,avgDDTT = metaPath_DT_Tsim(Tsim,avgDDT,3)
    
    return sumDDTT,maxDDTT
#-------------------------------------------------------------

def metaPath_DTDT(DT):
    
    TD = np.transpose(DT)
    DD = DT.dot(TD)

    sumM, maxM = metaPath_Dsim_DT(DD,DT,3)
    
    return sumM,maxM
#----------------------------------------------------------------