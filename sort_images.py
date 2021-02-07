import pickle
import numpy as np
import networkx as nx
import os
import shutil
import argparse


from CW import draw_graph,chinese_whispers

def image_sorter(G):
    """copies images from the source and pastes them to a directory.
    Each sub directory represents a pseudo class which contains images of the pseudo class assigned by
    the clustering algorithm

    Args:
        graph : networkx graph on which the clustering algorithm has been done on
    """
    root = "Sorted-pictures-test"
    if not os.path.exists(root):
        os.mkdir(root)

    for node,attribute in G.nodes.items():
        source = attribute["path"]
        destination = os.path.join(root,str(attribute["pseudoClass"]))

        if not os.path.exists(os.path.join(root,str(attribute["pseudoClass"]))):
            os.mkdir(os.path.join(root,str(attribute["pseudoClass"])))
         
        shutil.copy(source,destination)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t","--threshold",type=float,required=False,default=0.83,help="minimum  distance required between face embeddings to form a edge")
    
    parser.add_argument(
        "-itr","--iterations",type=int,required=False,default=30,help="number of iterations for the Chinese Whispers algorithm")

    args = vars(parser.parse_args())

    #Load the embeddings
    data = pickle.load(open("embeddings_test.pickle","rb"))
    
    # Draw the initial graph
    graph = draw_graph(data,args["threshold"])
    # Run the clustering algorithm on the graph
    graph = chinese_whispers(graph,args["iterations"])
    # Sort the images using the clusters
    image_sorter(graph)