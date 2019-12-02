'''
*******************************************************
1- Get the dataset type (defualt=nr dataset)
2- Read dataset with all needed information
3- generate all (drug, target) pairs with their labels
4- Read the similarity matrices
*******************************************************
'''
import argparse
import numpy as np
import collections
#-----------------------------------------

def parse_args():

	parser = argparse.ArgumentParser(description="Run DTIs code")
	parser.add_argument('--data', type=str, default='nr',  help='choose one of the datasets nr,gpcr, ic, e')

	return parser.parse_args()
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

	# to remove duplicate elements       
	targets = sorted(list(set(targets)))
	drugs = sorted(list(set(drugs)))
	DTIs = np.array(DTIs, dtype=np.float64)

	print('Number of drugs:',len(drugs))
	print('Number of targets:', len(targets))

	return drugs, targets, DTIs
#-------------------------------------------------------------------------

def built_multiple_similarity_matrix(sim_files,Dtype, data, length ):
    
    SimF = np.loadtxt(sim_files, delimiter='\n',dtype=str ,skiprows=0)
    Sim = []
    for i in range(0,len(SimF)):
        simMat = 'Input/'+data+'/'+Dtype+'sim/'+str(SimF[i]) 
        Sim.append(np.loadtxt(simMat, delimiter='\t',dtype=np.float64,skiprows=1,usecols=range(1,length+1)))
        
    return Sim
#---------------------------------------------------------------------------

def load_datasets(data):
	
	# read the interaction matrix
	DrugTargetF = "Input/"+data+"/"+data+"_admat_dgc.txt"
	DrugTarget = np.genfromtxt(DrugTargetF, delimiter='\t',dtype=str)

	# get all drugs and targets names with order preserving
	all_drugs, all_targets, DTIs = get_drugs_targets_names(DrugTarget)

	# read all files of similarties
	Tsim_files = 'Input/'+data+'/Tsim/selected_Tsim_files.txt'
	Dsim_files  = 'Input/'+data+'/Dsim/selected_Dsim_files.txt'

	# built the similarity matrices of multiple similarities
	D_sim = built_multiple_similarity_matrix(Dsim_files, 'D', data, len(all_drugs))
	T_sim = built_multiple_similarity_matrix(Tsim_files, 'T', data, len(all_targets))

	## Create R (drug, target, label) with known and unknown interaction
	R = tree()

	# Get all postive drug target interaction R
	with open('Input/'+data+'/R_'+data+'.txt','r') as f:
	    for lines in f:
	        line = lines.split()
	        line[0]= line[0].replace(":","")
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

    # prepare X = all (dr, tr) pairs, Y = labels
	X = np.asarray(pairX)
	Y = np.asarray(label)
	print('dimensions of all pairs', X.shape)

	return all_drugs, all_targets, D_sim, T_sim, DTIs, R, X, Y
#----------------------------------------------------------------------------------------