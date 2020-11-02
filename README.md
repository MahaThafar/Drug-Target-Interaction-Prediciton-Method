# Drug-Target-Interaction-Prediciton-Method
 
#### This repositery provides an implementation of DTiGEMS+ tool  which located in the folder (DTiGEMS+) and described in a research paper:
(Published in Journal of Cheminformatics 29 June 2020):

> DTiGEMS+: a network-based method for computational Drug-Target Interaction prediction using graph embedding, graph mining, and similarity-based techniques

Everything about the source code usage is explained in ReadME.md file inside the folder DTiGEMS+
https://github.com/MahaThafar/Drug-Target-Interaction-Prediciton-Method/tree/master/DTiGEMS%2B

---

#### The repositery also provides an example of node2vec code implemented inside the DTIs prediction code in the folder (DTIs_node2vec)

 ***About the folder (DTIs_node2vec):***
 - This example is applied on ion channel dataset (ic)
 - The code uses DTIs training part with single DD similarity and single TT similarity, combine them as edgelist (graph) and feeds them into node2vec model
 - embeddings will be generated for each node in the same code, and the rest of the code is similar to DTiGEMS+ model.
 - To run this code:
```
python DTIs_Main.py
```
- You can also provide some node2vec parameters when you run the code such as:
```
python DTIs_Main.py --dimension 32 --p 0.25 --q 2 --walk-length 30
```
 
 #### *Note:*
 >  When you run the code the AUPR result could be a little bit different than the other code (DTIs_Main_ic.py) because of randomness in node2vec when generates the embedding
 

---

#### For original node2vec code to generate new embeddings instead of reading generated embedding you can visit:

(all details to run the code as well as required parameters are provided with node2vec source code)

https://github.com/aditya-grover/node2vec

---

### IF you use any part of this code please cite:
Thafar, M.A., Olayan, R.S., Ashoor, H. et al. DTiGEMS+: drug–target interaction prediction using graph embedding, graph mining, and similarity-based techniques. J Cheminform 12, 44 (2020). https://doi.org/10.1186/s13321-020-00447-2
