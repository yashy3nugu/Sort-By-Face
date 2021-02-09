import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import argparse
import sys
import shutil
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool, cpu_count
from math import ceil
from facenet import compute_embedding
from tensorflow.keras.models import load_model
# Prevent tensorflow from logging INFO logs



def get_image_paths(root_dir):
    """Generates a list of paths for the images in a root directory
    Args:
        root_dir : string containing the relative path to root directory
    """
    paths = []
    for rootDir, directory, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
                paths.append(os.path.join(rootDir, filename))

    return paths


def save_embeddings(pool_data):
    """Function used by each processing pool to compute embeddings for part of a dataset

    Args:
        split_data : 
    """
    model = load_model("Models/facenet_keras.h5")
    output = []

    for count, path in enumerate(pool_data['imagePaths']):
        embeddings = compute_embedding(path,model)

        # in case no faces are detected
        if embeddings is None:
            continue

        if embeddings.shape[0] > 1:
            for embedding in embeddings:
                output.append({"path": path, "embedding": embedding})
        else:
            output.append({"path": path, "embedding": embeddings[0]})
        finished = (count/len(pool_data['imagePaths']))*100
        print("Finished {} %".format(finished))

    with open(pool_data['tempPath'], "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-src","--source", required=True, help="Path of the root directory where images are stored")
    
    parser.add_argument(
        "--processes", required=False, type=int, default=cpu_count(),help="Number of cores to be used to compute embeddings"
    )

    args = vars(parser.parse_args())

    image_paths = get_image_paths(args['source'])

    print("Found {} images..".format(len(image_paths)))

    # Define the number of processes to be used by the pool
    # Each process takes one core in the CPU
    PROCESSES = args["processes"]

    if PROCESSES > cpu_count():
        print("Number of processes greater than system capacity..")
        print("Defaulting to {} parallel processes".format(cpu_count()))
        PROCESSES = cpu_count()
    
    IMGS_PER_PROCESS = ceil(len(image_paths)/PROCESSES)

    # Split the images into equal sized batches for each process
    # Since we only need the embeddings for all the images the data can be split
    # into equal sized batches and each process can then independently compute the embeddings for the images.
    # The embeddings can then be concatenated after all of them are finished

    split_paths = []
    for i in range(0, len(image_paths), IMGS_PER_PROCESS):
        split_paths.append(image_paths[i:i+IMGS_PER_PROCESS])

    # Each process saves the embeddings computed by it into a pickle file in a temporary folder.
    # The temporary pickle files can then be loaded and we can generate a single pickle file containing all the embeddings for our data
    if not os.path.exists("temp"):
        os.mkdir("temp")

    split_data = []
    for pool_id, batch in enumerate(split_paths):
        temp_path = os.path.join("temp", "pool_{}.pickle".format(pool_id))

        pool_data = {
            "poolId": pool_id,
            "imagePaths": batch,
            "tempPath": temp_path
        }

        split_data.append(pool_data)

    # Create a pool which can execute more than one process paralelly
    pool = Pool(processes=PROCESSES)

    # Map the function
    print("Started {} processes..".format(PROCESSES))
    pool.map(save_embeddings, split_data)

    # Wait until all parallel processes are done and then execute main script
    pool.close()
    pool.join()

    concat_embeddings = []

    for filename in os.listdir("temp"):
        data = pickle.load(open(os.path.join("temp", filename), "rb"))

        for dictionary in data:
            concat_embeddings.append(dictionary)

    with open("embeddings_latest.pickle", "wb") as f:
        pickle.dump(concat_embeddings, f)

    shutil.rmtree("temp")
    print("Embeddings saved to disk.")
    # By now a single pickle file is created and the temporary files are deleted
