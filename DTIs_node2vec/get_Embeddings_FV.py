'''
************************************************************
This function is to get the exisitng generated embedding of node2vec.. The embedding are generated from the training dataset.
Same seed of cross validation is used to match he training and testing data of both embedding and classification..

allD and allT are lists of all drugs and targets (from similarity matrices)

The function recieve the fold numbers to read the match fold embedding, drugs and targets lists, drug and target dictionaries 
and return the feature vector for each drug and each target..
***********************************************************
'''

import numpy as np

def get_FV_drug_target(args,foldCounter,allT,allD):
    
    # Working with feature vector
    targets= {}
    drugs = {}
    n2v_Tr = []
    n2v_Dr = []

    directoryName = 'EMBED/ic/EmbeddingFold_'+str(foldCounter)+'.txt'
    fe = open(directoryName, 'w')
    count = 0
    ## ReadDT feature vectore that came after applying n2v on allGraph including just R_train part
    with open(args.output,'r') as f:
        line =f.readline()# to get rid of the sizes
        for line in f:
            fe.write(line) #to save fold embedding
            line = line.split()
            line[0]= line[0].replace(":","")
            # take the protien name as key (like dictionary)
            key = line[0]
            # remove the protien name to take the remaining 128 features
            line.pop(0)
            if key in allT:
                n2v_Tr.append(key) # to save all target as the same order of targets after applied n2v
                targets[key] = line
                count += 1
            else:
            #key in allD and its feature:  
                n2v_Dr.append(key) # to save all target as the same order of targets after applied n2v
                drugs[key] = line
                count += 1
                
    ### Create FV for drugs and for targets
    FV_drugs = []
    FV_targets = []

    for t in allT:
        FV_targets.append(targets[t])
    for d in allD:
        FV_drugs.append(drugs[d])

    FV_targets = np.array(FV_targets, dtype = float)
    FV_drugs = np.array(FV_drugs, dtype = float)     
    
    return FV_targets, FV_drugs
#---------------------------------------------------------------------------