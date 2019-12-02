# Drug-Target-Interaction-Prediciton-Method
 
#### This repositery provides an implementation of DTiGEMS+ tool  which located in the folder (DTiGEMS+) and described in a research paper:
(To be published):

> DTiGEMS+: a network-based method for computational Drug-Target Interaction prediction using graph embedding, graph mining, and similarity-based techniques

Everything about the source code usage is explained in ReadME.md file inside the folder

---

#### The repositery also provides an example of node2vec code implemented inside the DTIs prediction code in the folder (DTIs_node2vec)

 ***About the folder (DTIs_node2vec):***
 - This example is applied on ion channel dataset (ic)
 - The code uses DTIs training part with single DD similarity and single TT similarity, combine them as edgelist (graph) and feeds them into node2vec model
 - embeddings will be generated for each node in the same code, and the rest of the code is similar to DTiGEMS+ tool..
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
