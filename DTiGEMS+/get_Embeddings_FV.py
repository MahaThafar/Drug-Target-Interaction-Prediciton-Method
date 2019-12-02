'''
************************************************************
This function is to get the exisitng generated embedding for each dataset
after applying node2vec on the full graph G consists of (training part of DTIs, KNN DDsim, KNN TTsim)
Same seed of cross validation is used to match he training and testing data of 
both genertaing embedding code and classification (main) code..
allD and allT are lists of all drugs and targets (from similarity matrices)
The function recieve the fold numbers to read the match fold embedding, drugs and targets lists, 
drug and target dictionaries 
and return the feature vector for each drug and each target..
***********************************************************
'''
import numpy as np

#---------------------------------------------------------------------------
def get_FV_drug_target(foldCounter,allT,allD, data):
    # Working with feature vector
    targets ={}
    drugs ={}
    fileName = 'EMBED/'+data+'/EmbeddingFold_'+str(foldCounter)+'.txt'

    ## ReadDT feature vectore that came after applying n2v on allGraph including just R_train part
    with open(fileName,'r') as f:
        #line =f.readline()# to get rid of the sizes
        for line in f:
            line = line.split()
            line[0]= line[0].replace(":","")
            # take the protien name as key (like dictionary)
            key = line[0]
            # remove the protien name to take the remaining 128 features
            line.pop(0)
            if key in allT:
                targets[key] = line
            else:
            #key in allD and its feature: 
                drugs[key] = line
                
    ### Create FV for drugs and for targets
    FV_drugs = []
    FV_targets = []

    for t in allT:
        FV_targets.append(targets[t])

    for d in allD:
        FV_drugs.append(drugs[d])  

    # drug node2vec FV, and target node2vec FV
    FV_targets = np.array(FV_targets, dtype = float)
    FV_drugs = np.array(FV_drugs, dtype = float)     
    
    return FV_targets, FV_drugs
#------------------------------------------------------------------------------------