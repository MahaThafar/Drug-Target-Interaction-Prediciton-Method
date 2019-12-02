# coding: utf-8
# All needed packages
import argparse
import pandas as pd
import math as math
import numpy as np
import csv
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MaxAbsScaler

# Import my files
from load_datasets import *
from pathScores_functions import *
from get_Embeddings_FV import *
from training_functions import *
from GIP import *
from snf_code import *
######################################## START MAIN #########################################
#############################################################################################
def main():
    # get the parameters from the user
    args = parse_args()
    ## get the start time to report the running time
    t1 = time.time()

    ### Load the input data - return all pairs(X) and its labels (Y)..
    allD, allT, allDsim, allTsim, DrTr, R, X, Y = load_datasets(args.data)

    # create 2 dictionaries for drugs. the keys are their order numbers
    drugID = dict([(d, i) for i, d in enumerate(allD)])
    targetID = dict([(t, i) for i, t in enumerate(allT)])
    #-----------------------------------------
    ###### Define different classifiers
    # 1-Random Forest
    rf = RandomForestClassifier(n_estimators=200 ,n_jobs=10,random_state= 55,class_weight='balanced', criterion='gini') 
    
    # 2-Neural Network
    NN = MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2),random_state=1)

    # 3-Adaboost classifier
    ab = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=2,
                        min_samples_leaf=1, max_features='auto', random_state=10, max_leaf_nodes=None, 
                        class_weight= 'balanced'), algorithm="SAMME", n_estimators=90,random_state=32)
    #________________________________________________________________
    # 10-folds Cross Validation...............
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 22)
    skf.get_n_splits(X, Y)
    foldCounter = 1
    # all evaluation lists
    correct_classified = []
    ps = []
    recall = []
    roc_auc = []
    average_precision = []
    f1 = []
    Pre = []
    Rec = []
    AUPR_TEST = []
    TN = []
    FP = []
    FN = []
    TP = []
    all_dt_PredictedScore = []

    #Create two files to write the novel interactions based on predicted scores
    all_rankedPair_file = 'Novel_DTIs/'+args.data+'/'+args.data+'_all_Ranked_pairs.csv'
    novel_DT_file = 'Novel_DTIs/'+args.data+'/'+args.data+'_top_novel_DTIs.csv'

    # Start training and testing
    for train_index, test_index in  skf.split(X,Y):

        print("*** Working with Fold %i :***" %foldCounter)
        
        #first thing with R train to remove all edges in test (use it when finding path)
        train_DT_Matrix = Mask_test_index(test_index, X, Y, DrTr, drugID, targetID)
        DrTr_train = train_DT_Matrix.transpose()

        # get GIP Similarity from training known interactions, TT sim, DD sim
        DT_impute_D = impute_zeros(DrTr_train,allDsim[0])
        DT_impute_T = impute_zeros(np.transpose(DrTr_train),allTsim[0])
        GIP_D = Get_GIP_profile(np.transpose(DT_impute_D),"d")
        GIP_T = Get_GIP_profile(DT_impute_T,"t")
        #--------------------------------------------

        DDsim = []
        TTsim = []

        for sim in allDsim:
            DDsim.append(sim)
        #DDsim.append(GIP_D)

        for sim in allTsim:
            TTsim.append(sim)
        #TTsim.append(GIP_T)

        fused_simDr = SNF(DDsim,K=5,t=3,alpha=1.0)
        fused_simTr = SNF(TTsim,K=5,t=3,alpha=1.0)
        ##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # insert node2vec code here to generate embedding in the same code.....
        #------------------------------ node2vec ------------------------------

        targetFV, drugFV = get_FV_drug_target(foldCounter, allT, allD, args.data)
        
        # Calculate cosine similarity for each drug pair, and for each target pair
        cos_simDD = Cosine_Similarity(drugFV)
        cos_simTT = Cosine_Similarity(targetFV)
        # normalize simiarities to be in positive range [0,1]
        cos_simDD = normalizedMatrix(cos_simDD)
        cos_simTT  = normalizedMatrix(cos_simTT )
        #--------------------------------------------------------------------- 

        # Generate all featres from the matrix multiplication of each path strucutre
        # list for each feature (Graph G1)
        sumDDD, maxDDD = DDD_TTT_sim(fused_simDr)
        sumTTT, maxTTT = DDD_TTT_sim(fused_simTr)
        
        sumDDT,maxDDT = metaPath_Dsim_DT(fused_simDr,DrTr_train,2) 
        sumDTT,maxDTT = metaPath_DT_Tsim(fused_simTr,DrTr_train,2)

        sumDDDT,_= metaPath_Dsim_DT(sumDDD,DrTr_train,3)
        _,maxDDDT = metaPath_Dsim_DT(maxDDD,DrTr_train,3)

        sumDTTT,_ = metaPath_DT_Tsim(sumTTT,DrTr_train,3)
        _,maxDTTT = metaPath_DT_Tsim(maxTTT,DrTr_train,3)

        sumDTDT,maxDTDT = metaPath_DTDT(DrTr_train)
        sumDDTT,maxDDTT = metaPath_DDTT(DrTr_train,fused_simDr,fused_simTr)
    #============================================================================== 
        # Generate all featres from the matrix multiplication of each path strucutre
        # list for each feature (Graph G2)
        sumDDD2, maxDDD2 = DDD_TTT_sim(cos_simDD)
        sumTTT2, maxTTT2 = DDD_TTT_sim(cos_simTT)
        
        sumDDT2,maxDDT2 = metaPath_Dsim_DT(cos_simDD,DrTr_train,2) 
        sumDTT2,maxDTT2 = metaPath_DT_Tsim(cos_simTT,DrTr_train,2)

        sumDDDT2,_ = metaPath_Dsim_DT(sumDDD2,DrTr_train,3)
        _,maxDDDT2 = metaPath_Dsim_DT(maxDDD2,DrTr_train,3)

        sumDTTT2,_ = metaPath_DT_Tsim(sumTTT2,DrTr_train,3)
        _,maxDTTT2 = metaPath_DT_Tsim(maxTTT2,DrTr_train,3)

        sumDTDT2,maxDTDT2 = metaPath_DTDT(DrTr_train)
        sumDDTT2,maxDDTT2 = metaPath_DDTT(DrTr_train,cos_simDD,cos_simTT)
    #==============================================================================  
    ### Build feature vector and class labels
        DT_score = []
        for i in range(len(allD)):
            for j in range(len(allT)):        
                pair_scores = (allD[i], allT[j],\
                            # path scores from G1
                               sumDDT[i][j],sumDDDT[i][j],\
                               sumDTT[i][j],sumDTTT[i][j], sumDDTT[i][j], sumDTDT[i][j],\
                               maxDDT[i][j],maxDDDT[i][j], \
                               maxDTT[i][j],maxDTTT[i][j],maxDDTT[i][j],maxDTDT[i][j],\
                            # path scores from G2
                               sumDDT2[i][j],sumDDDT2[i][j],\
                               sumDTT2[i][j],sumDTTT2[i][j], sumDDTT2[i][j], sumDTDT2[i][j],\
                               maxDDT2[i][j],maxDDDT2[i][j], \
                               maxDTT2[i][j],maxDTTT2[i][j],maxDDTT2[i][j],maxDTDT2[i][j])
                DT_score.append(pair_scores)
        
        features = []
        class_labels = []
        DT_pair = []
        # Build the feature vector - Concatenate features from G1,G2
        for i in range(len(DT_score)):
            dr = DT_score[i][0]
            tr = DT_score[i][1] 
            edgeScore = DT_score[i][2], DT_score[i][3], DT_score[i][4],DT_score[i][5],\
                        DT_score[i][8],DT_score[i][9], DT_score[i][10], DT_score[i][11],\
                        DT_score[i][14], DT_score[i][15],DT_score[i][16],DT_score[i][17],DT_score[i][18],\
                        DT_score[i][20], DT_score[i][21],DT_score[i][22]
           
            dt = DT_score[i][0], DT_score[i][1]
            DT_pair.append(dt)
            features.append(edgeScore)
            # same label as the begining
            label = R[dr][tr]
            class_labels.append(label)

        ## Start Classification Task
        # featureVector and labels for each pair
        XX = np.asarray(features)
        YY = np.array(class_labels)

        #Apply normalization using MaxAbsolute normlization
        max_abs_scaler = MaxAbsScaler()
        X_train = max_abs_scaler.fit(XX[train_index]) 
        X_train_transform = X_train.transform(XX[train_index])

        X_test_transform = max_abs_scaler.transform(XX[test_index])

        # Apply different oversampling techniques:
        # ros = RandomOverSampler(random_state= 10)
        sm = SMOTE(random_state=10)
        # ada = ADASYN(random_state=10)
        X_res, y_res= sm.fit_sample(X_train_transform, YY[train_index])
        
        # fit the model
        NN.fit(X_res, y_res)
        predictedClass = NN.predict(X_test_transform)
        predictedScore = NN.predict_proba(X_test_transform)[:, 1]

        #Find the novel interactions based on predicted scores
        fold_dt_score = []
        for idx, c in zip(test_index,range(0,len(predictedScore))):
            # write drug, target, predicted score of class1, predicted class, actual class
            dtSCORE = str(DT_pair[idx]),predictedScore[c],predictedClass[c],YY[idx]
            all_dt_PredictedScore.append(dtSCORE)

        # ------------------- Print Evaluation metrics for each fold --------------------------------
        print("@@ Validation and evaluation of fold %i @@" %foldCounter)
        print(YY[test_index].shape, predictedClass.shape)

        cm = confusion_matrix(YY[test_index], predictedClass)
        TN.append(cm[0][0])
        FP.append(cm[0][1])
        FN.append(cm[1][0])
        TP.append(cm[1][1])
        print("Confusion Matrix for this fold")
        print(cm)

        print("Correctly Classified Instances: %d" %accuracy_score(Y[test_index], predictedClass, normalize=False))
        correct_classified.append(accuracy_score(Y[test_index], predictedClass, normalize=False))

        #print("Precision Score: %f" %precision_score(Y[test_index], predictedClass))
        ps.append(precision_score(Y[test_index], predictedClass,average='weighted'))

        #print("Recall Score: %f" %recall_score(Y[test_index], predictedClass)
        recall.append(recall_score(Y[test_index], predictedClass, average='weighted'))

        print("F1 Score: %f" %f1_score(Y[test_index], predictedClass, average='weighted'))
        f1.append(f1_score(Y[test_index], predictedClass,average='weighted'))

        print("Area ROC: %f" %roc_auc_score(Y[test_index], predictedScore))
        roc_auc.append(roc_auc_score(Y[test_index], predictedScore))

        p, r, _ = precision_recall_curve(Y[test_index],predictedScore,pos_label=1)
        aupr = auc(r, p)
        print("AUPR auc(r,p) = %f" %aupr)
        AUPR_TEST.append(aupr)

        Pre.append(p.mean())
        Rec.append(r.mean())
        average_precision.append(average_precision_score(Y[test_index], predictedScore))

        print(classification_report(Y[test_index], predictedClass))
        print('--------------------------------------------------')
        foldCounter += 1
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

    # Write predicted scores into file to find novel interactions:
    dt_df = pd.DataFrame(all_dt_PredictedScore, columns=['DT_pair', 'Predicted_score_class1', 'Predicted_Class', 'Actual_Class'])
    dt_df = dt_df.sort_values(by='Predicted_score_class1', ascending=False)
    
    dt_df = dt_df[dt_df['Predicted_Class']==1]
    novel_dt = dt_df[dt_df['Actual_Class']==0]

    dt_df.to_csv(all_rankedPair_file, sep='\t', index=None)
    novel_dt.to_csv(novel_DT_file,sep='\t', index=None)
    #--------------------------------------------------------------------
    ############# Evaluation Metrics ####################################
    # Confusion matrix for all folds
    ConfMx = np.zeros((cm.shape[0],cm.shape[0]))
    ConfMx[0][0] = str( np.array(TN).sum() )
    ConfMx[0][1] = str( np.array(FP).sum() )
    ConfMx[1][0] = str( np.array(FN).sum() )
    ConfMx[1][1] = str( np.array(TP).sum() )

    ### Print Evaluation Metrics.......................
    print("Result(Correct_classified): " + str( np.array(correct_classified).sum() ))
    print("Results:precision_score = " + str( np.array(ps).mean().round(decimals=3) ))
    print("Results:recall_score = " + str( np.array(recall).mean().round(decimals=3) ))
    print("Results:f1 = " + str( np.array(f1).mean().round(decimals=3) ))
    print("Results:roc_auc = " + str( np.array(roc_auc).mean().round(decimals=3) ))
    print("Results: AUPR on Testing auc(r,p) = " + str( np.array(AUPR_TEST).mean().round(decimals=3)))
    print("Confusion matrix for all folds")
    print(ConfMx) 
    print('_____________________________________________________________')
    print('Running Time for the whole code:', time.time()-t1)  
    print('_____________________________________________________________')  
#####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    main()
#####-------------------------------------------------------------------------------------------------------------
####################### END OF THE CODE ##########################################################################