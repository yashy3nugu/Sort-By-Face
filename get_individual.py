import pickle
import shutil
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from clusterer import draw_graph, chineseWhispers
from facenet import load_and_align, compute_embedding


def get_person(graph,user_node,destination):
    """Copies all the images of a cluster to the destination folder

    Args:
        graph : networkx graph on which the clustering algorithm has been done
        user_node : node pertaining to the user's image in the parameter graph
    """
    user_cluster = graph.nodes[user_node]['pseudoClass']
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    for node,attribute in graph.nodes.items():
        if attribute['pseudoClass'] == user_cluster:
            shutil.copy(attribute['path'],destination)
    print("Your images have been copied to the folder {}".format(destination))
        
         

# Prevent tensorflow from logging INFO logs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-src","--source",required=True,help="path for the image of the person you want to get images of")
    
    parser.add_argument(
        "-dest","--destination",required=True,help="path for the folder where you want to store images")
    
    parser.add_argument(
        "-t","--threshold",type=float,required=False,default=0.83,help="minimum  distance required between face embeddings to form a edge")
    
    parser.add_argument(
        "-itr","--iterations",type=int,required=False,default=30,help="number of iterations for the Chinese Whispers algorithm")

    args = vars(parser.parse_args())

    # load the user's image and compute embedding
    model = load_model("Weights/facenet_keras.h5")
    user_embedding = compute_embedding(args["source"],model)
    print(user_embedding.shape)

    # Load the embeddings from the corpus
    data = pickle.load(open("embeddings.pickle","rb"))

    # We will first assign a node to the user for the graph used in the clustering algorithm
    # After running the clustering algorithm, since we know the node the user's image's embedding is in
    # We can then check the cluster the node was assigned to and then copy all the images in the same cluster
    user_node = len(data) + 1
    data.append({"path":args["source"],"embedding":user_embedding[0]})

    graph = draw_graph(data,args["threshold"])
    graph = chineseWhispers(graph,args["iterations"])

    # Copy the respective images
    get_person(graph,user_node,args["destination"])



