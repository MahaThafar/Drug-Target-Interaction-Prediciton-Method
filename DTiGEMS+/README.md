# DTiGEMS+: Compuational Method for Drug-Target Interactions Prediction using Graph Embedding, Graph Mining, and Similarity-based Techniques

- Received                      Accepted                  Published
- 10 December 2019              16 June 2020              29 June 2020


----
This code is implemented using Python 2.7.9, but since Python 2.7 will no longer supported after January 2020, It is upgraded to be run using Python 3.6

For any qutions please contact the first author:


  Maha Thafar

Email: maha.thafar@kaust.edu.sa

Computer, Electrical and Mathematical Sciences and Engineering Division (CEMSE), Computational Bioscience Research Center, Computer (CBRC), King Abdullah University of Science and Technology (KAUST).

----

## Getting Started

### Prerequisites:

There are several required Python packages to run the code:
- gensim (for node2vec code)
- numpy
- Scikit-learn
- imblearn
- pandas

These packages can be installed using pip or conda as the follwoing example
```
pip install -r requirements.txt
```
----

### Files Description:
#### *There are Three folders:*

  **1.Input folder:** 
  that includes four folder for 4 datasets include: 
   - Nuclear Receptor dataset (nr),
   - G-protein-coupled receptor (gpcr),
   - Ion Channel (IC), 
   - Enzyme (e)
     which each one of them has all required data of drug-target interactions (in Adjacency matrix and edgelist format) and drug-drug and target-target similarities in (square matrix format) - all matrices include items names.
  
  **2.Embedding folder:**
  that has also four folders coressponding for four datasets,
     each folder contains the generated node2vec embeddings files for each fold of training data
     
  **3.Novel_DTIs folder:**
  that has also four folders coressponding for four datasets, 
     to write the novel DTIs (you should create directory for each dataset)
  
---
#### *There are 10 files:*
(Four main functions, one main for each dataset, and the other functions are same for all datasets which are imported in each main function)

- **load_datasets.py** --> to read the input data including interactions and similarities
- **get_Embedding_FV.py** --> to read the node2vec generated embedding and get the FV for each drug and target (CV random seed=22)
- **training_functions.py** --> for several training and processing functions such as edgeList, Cosine_similarity, ..
- **pathScores.py** --> to calculate and return all path scores for 6 path structures
- **snf_code.py** --> Similarity Network Fusion functions
- **GIP.py** --> to calculate and return gussian interaction profile similarity

- **Four main functions**
one for each dataset:
> - DTIs_Main_nr.py
> - DTIs_Main_gpcr.py
> - DTIs_Main_ic.py
> - DTIs_Main_enzyme.py

---
## Installing:

To get the development environment runining, the code get one parameter from the user which is the dataset name (the defual dataset is nr)
run:

```
python DTIs_Main_nr.py --data nr
```
```
python DTIs_Main_gpcr.py --data gpcr
```
```
python DTIs_Main_ic.py --data ic
```
```
python DTIs_Main_e.py --data e
```

------------------
## For citations:
```
Thafar, M.A., Olayan, R.S., Ashoor, H. et al. DTiGEMS+: drug–target interaction prediction using graph embedding, graph mining, and similarity-based techniques. J Cheminform 12, 44 (2020). https://doi.org/10.1186/s13321-020-00447-2
```

