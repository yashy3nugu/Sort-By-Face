import pickle
import numpy as np
import networkx as nx
import os
import shutil
from random import shuffle

def chineseWhispers(data,threshold,iterations):

    nodes = []
    edges = []

    embeddings = np.array([d['embedding'] for d in data])

    # Iterate through  all embeddings computed from the corpus
    for index, embedding in enumerate(data):

        # current_node represents the unique number by which a node is identified
        current_node = index+1
        
        node = (current_node, {'pseudo_class':current_node,"path":data[index]['path']})
        nodes.append(node)


        

if __name__ == "__main__":
    data = pickle.load(open("embeddings.pickle","rb"))
    
    embeddings = np.array([d['embedding'] for d in data])

    G = nx.Graph()

    print(data)

