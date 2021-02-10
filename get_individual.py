import sys
import pickle
import shutil
import argparse
import numpy as np
import os
import logging


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


from tensorflow.keras.models import load_model
from CW import draw_graph, chinese_whispers
from facenet import load_and_align, compute_embedding


def get_person(graph,user_node,destination):
    """Copies all the images of a cluster (person) to the destination folder

    Args:
        graph : networkx graph on which the clustering algorithm has been done
        user_node : node pertaining to the user's image in the graph
    """
    user_cluster = graph.nodes[user_node]['cluster']
    user_path = graph.nodes[user_node]['source']
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    for node,attribute in graph.nodes.items():
        if (attribute['cluster'] == user_cluster) and (attribute['source'] != user_path):
            try:
                shutil.copy(attribute['source'],destination)
            except FileNotFoundError:
                pass

    print("Your images have been copied to the folder {}".format(destination))
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-src", "--source", required=True, help="path for the image of the person you want to get images of")
    
    parser.add_argument(
        "-dest", "--destination", required=True, help="path for the folder where you want to store images")
    
    parser.add_argument(
        "-t" ,"--threshold", type=float, required=False, default=0.67, help="minimum cosine similarity required between face embeddings to form a edge")
    
    parser.add_argument(
        "-itr", "--iterations", type=int, required=False, default=30, help="number of iterations for the Chinese Whispers algorithm")

    args = vars(parser.parse_args())

    # load the user's image and compute embedding
    model = load_model("Models/facenet_keras.h5")
    user_embedding = compute_embedding(args["source"],model)
    
    if user_embedding is None:
        # cv2.imread() returns a NoneType object instead of throwing an error for invalid image paths
        print("Image not found, please enter valid image path")
        sys.exit(1)
    elif user_embedding.shape[0] > 1:
        print("Found more than one face in picture. Please give a picture with only one face..")
        sys.exit(1)
    elif user_embedding.shape[0] == 0:
        print("Found no faces. Please give a picture with a face..")
        sys.exit(1)

    # Load the embeddings from the corpus
    data = pickle.load(open("embeddings.pickle","rb"))

    # We will first assign a node to the user for the graph used in the clustering algorithm
    # After running the clustering algorithm, since we know the node the user's image's embedding is in
    # We can then check the cluster the node was assigned to and then copy all the images in the same cluster
    user_node = len(data) + 1
    data.append({"path":args["source"],"embedding":user_embedding[0]})

    graph = draw_graph(data,args["threshold"])
    graph = chinese_whispers(graph,args["iterations"])

    # Copy the respective images
    get_person(graph,user_node,args["destination"])



