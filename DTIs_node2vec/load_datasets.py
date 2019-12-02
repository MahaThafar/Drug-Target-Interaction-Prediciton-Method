# All needed packages
import argparse
import numpy as np
import collections

#-----------------------------------------
def tree():
    return collections.defaultdict(tree)

#-----------------------------------------
def get_drugs_targets_names(DT):
	# remove the drugs and targets names from the matrix
	DTIs = np.zeros((DT.shape[0]-1,DT.shape[1]-1))

	drugs = []
	targets = []

	for i in range(1,DT.shape[0]):
	    for j in range(1,DT.shape[1]):
	        targets.append(DT[i][0])
	        drugs.append(DT[0][j])
	        DTIs[i-1][j-1] = DT[i][j]

	# In[41]:
	# to remove duplicate elements       
	targets = sorted(list(set(targets)))
	drugs = sorted(list(set(drugs)))

	print('# of drugs',len(drugs))
	print('# of targets', len(targets))

	return drugs, targets, DTIs

#-----------------------------------------
def built_multiple_similarity_matrix(sim_files,dtype, length ):
    
    SimF = np.loadtxt(sim_files, delimiter='\n',dtype=str ,skiprows=0)
    Sim = []
    for i in range(0,len(SimF)):
        simMat = 'Input/ic/'+dtype+'sim/'+str(SimF[i])
        Sim.append(np.loadtxt(simMat, delimiter='\t',dtype=np.float64,skiprows=1,usecols=range(1,length+1)))
        
    return Sim
#------------------------------------------------------

def load_datasets():
	# read the interaction matrix
	DrugTargetF = "Input/ic/ic_admat_dgc.txt"
	DrugTarget = np.genfromtxt(DrugTargetF, delimiter='\t',dtype=str)

	# read all files of similarties
	Tsim_files = 'Input/ic/Tsim/Tsim_files.txt'
	Dsim_files  = 'Input/ic/Dsim/Dsim_files.txt'

	# get all drugs and targets names with order preserving
	all_drugs, all_targets, DTIs = get_drugs_targets_names(DrugTarget)

	# built the similarity matrices of multiple similarities
	D_sim = built_multiple_similarity_matrix(Dsim_files, 'D',  len(all_drugs))
	T_sim = built_multiple_similarity_matrix(Tsim_files, 'T',  len(all_targets))

	## Create R (drug, target, label) with known and unknown interaction
	R = tree()

	# Get all postive drug target interaction R
	with open('Input/ic/R_ic.txt','r') as f:
	    for lines in f:
	        line = lines.split()
	        line[0]= line[0].replace(":","")
	        # put class label = +1 to all exit T_D in R in for D_T_label
	        R[line[1]][line[0]] = 1

	#######################################################################
	#build the BIG R with all possible pairs and assign labels
	label = []
	pairX = []
	for d in all_drugs:
		for t in all_targets:
			p = d, t
            # add negative label to non exit pair in R file
			if R[d][t] != 1:
				R[d][t] = 0
				l = 0
			else:
				l = 1

			label.append(l)
			pairX.append(p)

    # prepare X = pairs, Y = labels and build the random forest model
	X = np.asarray(pairX) 
	Y = np.asarray(label)

	return all_drugs, all_targets, D_sim, T_sim, DTIs, R, X, Y
#----------------------------------------------------------------------------------------