import sys
import pickle
import shutil
import argparse
import cv2
import numpy as np
import os
import logging

# Prevent tensorflow from logging in multiple processes simultaneously
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool, cpu_count
from math import ceil
from facenet import compute_embedding
from tensorflow.keras.models import load_model
from tqdm import tqdm



def get_image_paths(root_dir):
    """Generates a list of paths for the images in a root directory and ignores rest of the files
    Args:
        root_dir : string containing the relative path to root directory
    Returns:
        paths : list containing paths of the images in the directory
    """
    if not os.path.exists(root_dir):
        print("Directory not found, please enter valid directory..")
        sys.exit(1)

    paths = []
    for rootDir, directory, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                paths.append(os.path.join(rootDir, filename))

    return paths


def save_embeddings(process_data):
    """Function used by each processing pool to compute embeddings for part of a dataset

    Args:
        process_data : dictionary consisting of data to be used by the pool 
    """
    # load the model for each process
    model = load_model("Models/facenet_keras.h5")
    # progress bar to track
    bar = tqdm(total=len(process_data['image_paths']),position=process_data['process_id'])

    output = []
    for count, path in enumerate(process_data['image_paths']):
        embeddings = compute_embedding(path,model)

        # in case no faces are detected
        if embeddings is None:
            continue
        
        # multiple faces in an image
        if embeddings.shape[0] > 1:
            for embedding in embeddings:
                output.append({"path": path, "embedding": embedding})
        else:
            output.append({"path": path, "embedding": embeddings[0]})
        bar.update()

    bar.close()
    bar.clear()
    # write the embeddings computed by a process into the temporary folder
    with open(process_data['temp_path'], "wb") as f:
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

    if len(image_paths) == 0:
        print("Found 0 images. Please enter a directory with images..")
        sys.exit(1)

    print("Found {} images..".format(len(image_paths)))

    # Define the number of processes to be used by the pool
    # Each process takes one core in the CPU
    processes = args["processes"]

    if processes > cpu_count():
        print("Number of processes greater than system capacity..")
        processes = cpu_count()
        print("Defaulting to {} parallel processes..".format(processes))
    
    imgs_per_process = ceil(len(image_paths)/processes)

    # Split the images into equal sized batches for each process
    # Since we only need the embeddings for all the images the data can be split
    # into equal sized batches and each process can then independently compute the embeddings for the images.
    # The embeddings can then be concatenated after all of them are finished

    split_paths = []
    for i in range(0, len(image_paths), imgs_per_process):
        split_paths.append(image_paths[i:i+imgs_per_process])

    # Each process saves the embeddings computed by it into a pickle file in a temporary folder.
    # The temporary pickle files can then be loaded and we can generate a single pickle file containing all the embeddings for our data
    if not os.path.exists("temp"):
        os.mkdir("temp")

    split_data = []
    for process_id, batch in enumerate(split_paths):
        temp_path = os.path.join("temp", "process_{}.pickle".format(process_id))

        process_data = {
            "process_id": process_id,
            "image_paths": batch,
            "temp_path": temp_path
        }

        split_data.append(process_data)

    # Create a pool which can execute more than one process paralelly
    pool = Pool(processes=processes)

    # Map the function
    print("Started {} processes..".format(processes))
    pool.map(save_embeddings, split_data)

    # Wait until all parallel processes are done and then execute main script
    pool.close()
    pool.join()

    # Once all processes are done load the pickle files in the temporary folder 'temp' and save them all in one file.
    # After that the temporary folder is deleted
    concat_embeddings = []

    for filename in os.listdir("temp"):
        data = pickle.load(open(os.path.join("temp", filename), "rb"))

        for dictionary in data:
            concat_embeddings.append(dictionary)

    with open("embeddings.pickle", "wb") as f:
        pickle.dump(concat_embeddings, f)

    shutil.rmtree("temp")
    print("Saved embeddings of {} faces to disk..".format(len(concat_embeddings)))
    # By now a single pickle file is created and the temporary files are deleted
