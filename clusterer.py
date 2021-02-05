import pickle
import numpy as np
import networkx as nx
import os
import shutil
from random import shuffle

def get_distances(embeddings,current_face):
    """Returns an array containing the euclidean distances between a given face's embedding and an
    of other face embeddings 

    Args:
        embeddings : numpy array consisting of the embeddings
        current_face : numpy array consisiting of embedding for a single face
    """
    # current_face is broadcasted to 0th axis of embeddings
    return np.linalg.norm(embeddings - current_face, axis=1)

def chineseWhispers(data,threshold,iterations):

    G = nx.Graph()
    # Lists used to store nodes and edges for a graph
    nodes = []
    edges = []

    embeddings = np.array([d['embedding'] for d in data])

    # Iterate through  all embeddings computed from the corpus
    for index, embedding in enumerate(data):

        # current_node represents the unique number by which a node is identified
        # Each face in the corpus is assigned to a node which contains the pseudo class and the path to the image containing the face.
        # If an image contains two or more faces the respective number of nodes corresponding to each face is initialized.
        # Initially all the faces are assigned to a seperate pseudo-class.
        # After a specific number of iterations the algorithm groups similar faces into the same pseudo-class.

        current_node = index+1
        
        node = (current_node, {'pseudoClass':current_node,"path":data[index]['path']})
        nodes.append(node)

        if current_node >= len(data):
            break

        # Get the euclidean distance for the face embedding of the current node and all the subsequent face embeddings
        emb_distances = get_distances(embeddings[index+1:])

        # list containing all the edges for current node
        current_node_edges = []

        # iterate through the euclidean distances  
        for i,weight in enumerate(emb_distances):
            
            # Add an edge between the current face embedding's node and the other face embedding's node
            # if the distance is lesser than threshold
            if weight < threshold:

                current_node_edges.append((current_node,current_node+i+1,{"weight":weight}))

        # Add the current edges to the list
        edges = edges + current_node_edges
   
    G.add_nodes_from(nodes)
    G.add_nodes_from(edges)

    for _ in range(iterations):
        # Get all the nodes of the graph and shuffle them
        nodes = list(G.nodes())
        shuffle(nodes)

        # Iterate through the shuffled nodes
        for node in nodes:
            # Get the neighbours for the node
            neighbours = G[node]

            pseudo_classes = {}
            #Firstly collect all the pseudo-classes the neighbours belong to
            for neighbour in neighbours:
                # For a given neighbour check the pseudo-class it belong to.
                # For the same key in the dictionary of the pseudo-class add the 
                # weight between the node and the current neighbour to the value of the key

                if G.nodes[neighbour]['pseudoClass'] in pseudo_classes:

                    pseudo_classes[G.nodes[neighbour]]
                    
                    clusters[G.nodes[neighbour]['pseudoClass']] += G[node][neighbour]['weight']
                else:
                    clusters[G.nodes[neighbour]['pseudoClass']] = G[node][neighbour]['weight']
                
                weight_sum = 0
                best_pseudo_class = None

            # The best pseudo-class for the particular node is then the
            # pseudo-class whose sum of edge weights to the node is maximum for the edges the node belongs to 
            for pseudo_class in pseudo_classes:
                if pseudo_classes[pseudo_class] >  weight_sum:
                    weight_sum = pseudo_classes[pseudo_class]
                    best_pseudo_class = pseudo_class

            G.nodes[node]['pseudoClass'] = best_pseudo_class

    return G


if __name__ == "__main__":
    data = pickle.load(open("embeddings.pickle","rb"))
    
    embeddings = np.array([d['embedding'] for d in data])

    G = nx.Graph()

    print(data)

