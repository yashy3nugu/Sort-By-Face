import pickle
import numpy as np
import networkx as nx
import os
import shutil
import argparse
from random import shuffle

def get_distances(embeddings,current_face):
    """Returns an array containing the euclidean distances between a given face's embedding and an
    of other face embeddings 

    Args:
        embeddings : numpy array consisting of the embeddings
        current_face : numpy array consisiting of embedding for a single face
    """
    # current_face is broadcasted to 0th axis of embeddings
    # To-do: Check with cosine similarity
    return np.linalg.norm(embeddings - current_face, axis=1)

def draw_graph(data,threshold):
    """Draws a networkx graph in which each node represents an image in the corpus.
    The attributes of the node contain the embedding of the image computed by the face descriptor
    and the cluster it belongs to. Initially all the images are given their own cluster

    Args:
        data : The list of dictionaries containing the embeddings. (obtained by running the script `embedder.py` and loading the pickle file generated)
        threshold : Minimum distance required between two face embeddings to form an edge between their respective nodes.

    Returns:
        G: the initial networkx graph required for the Chinese Whispers algorithm
    """
    G = nx.Graph()
    # Lists used to store nodes and edges for a graph
    G_nodes = []
    G_edges = []

    embeddings = np.array([dictionary['embedding'] for dictionary in data])

    # Iterate through  all embeddings computed from the corpus
    for index, embedding in enumerate(data):

        # current_node represents the unique number by which a node is identified
        # Each face in the corpus is assigned to a node which contains the pseudo class and the path to the image containing the face.
        # If an image contains two or more faces the respective number of nodes corresponding to each face is initialized.
        # Initially all the faces are assigned to a seperate pseudo-class.
        # After a specific number of iterations the algorithm groups similar faces into the same pseudo-class.

        current_node = index+1
        
        node = (current_node, {'pseudoClass':current_node,"path":data[index]['path']})
        G_nodes.append(node)

        if current_node >= len(data):
            break
        print("Calculating distances for node "+ str(current_node))
        # Get the euclidean distance for the face embedding of the current node and all the subsequent face embeddings
        # We only need to caluclate for the subsequent ones because we already calculated for previous ones in earlier iterations and the edges have already been formed
        emb_distances = get_distances(embeddings[index+1:],data[index]['embedding'])

        # list containing all the edges for current node
        current_node_edges = []

        # iterate through the euclidean distances  
        for i,weight in enumerate(emb_distances):
            
            # Add an edge between the current face embedding's node and the other face embedding's node
            # if the distance is lesser than threshold
            if weight < threshold:

                current_node_edges.append((current_node,current_node+i+1,{"weight":weight}))

        # Add the current edges to the list
        G_edges = G_edges + current_node_edges
   
    G.add_nodes_from(G_nodes)
    G.add_edges_from(G_edges)

    return G

def chineseWhispers(G,iterations):
    """Applies the Chinese Whispers algorithm to the graph

    Args:
        G : networkx graph to represent the face embeddings
        iterations : number of iterations for the algorithm

    Returns:
        G: networkx graph where the embeddings are clustered
    """

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
                    
                    pseudo_classes[G.nodes[neighbour]['pseudoClass']] += G[node][neighbour]['weight']
                else:
                    pseudo_classes[G.nodes[neighbour]['pseudoClass']] = G[node][neighbour]['weight']
                
            weight_sum = 0
            best_pseudo_class = None

            # The best pseudo-class for the particular node is then the
            # pseudo-class whose sum of edge weights to the node is maximum for the edges the node belongs to 
            for pseudo_class in pseudo_classes:
                if pseudo_classes[pseudo_class] >  weight_sum:
                    weight_sum = pseudo_classes[pseudo_class]
                    best_pseudo_class = pseudo_class

            # If there is only one image of a person then dont assign the pseudo-class
            if best_pseudo_class is None:
                continue

            G.nodes[node]['pseudoClass'] = best_pseudo_class

    return G

def image_sorter(G):
    """copies images from the source and pastes them to a directory.
    Each sub directory represents a pseudo class which contains images of the pseudo class assigned by
    the clustering algorithm

    Args:
        graph : networkx graph on which the clustering algorithm has been done on
    """
    root = "Sorted-pictures"
    if not os.path.exists(root):
        os.mkdir(root)

    for node,attribute in G.nodes.items():
        source = attribute["path"]
        destination = os.path.join(root,str(attribute["pseudoClass"]))

        if not os.path.exists(os.path.join(root,str(attribute["pseudoClass"]))):
            os.mkdir(os.path.join("Sorted-pictures",str(attribute["pseudoClass"])))
         
        shutil.copy(source,destination)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t","--threshold",type=float,required=True,help="minimum  distance required between face embeddings to form a edge")
    
    parser.add_argument(
        "-itr","--iterations",type=int,required=False,default=20,help="number of iterations for the Chinese Whispers algorithm")

    args = vars(parser.parse_args())

    #Load the embeddings
    data = pickle.load(open("embeddings.pickle","rb"))
    
    # Draw the initial graph
    graph = draw_graph(data,args["threshold"])
    # Run the clustering algorithm on the graph
    graph = chineseWhispers(graph,args["iterations"])
    # Sort the images using the clusters
    image_sorter(graph)


