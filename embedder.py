import multiprocessing
import cv2
import numpy as np
import os
import pickle
from math import ceil
# embedder
#from keras_facenet import FaceNet

from aligner import load_and_align

#embedder = FaceNet()

def get_image_paths(root_dir):
    """Generates a list of paths for the images in a root directory
    Args:
        root_dir : string containing the relative path to root directory
    """
    paths = []
    for rootDir,directory,filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
                paths.append(os.path.join(rootDir,filename))

    return paths


def compute_embedding(pool_data,detector="HOG"):
    """Function used by each processing pool to compute embeddings for part of a dataset

    Args:
        split_data : 
    """

    output = []

    for count,path in enumerate(pool_data['imagePaths']):
        image = load_and_align(path,detector)

        if image is None: # Case where no faces were found in image
            continue

        embeddings = embedder.embeddings(image)

        if embeddings.shape[0]>1:
            for embedding in embeddings:
                output.append({"path":path,"embedding":embedding})
        else:
            output.append({"path":path,"embedding":embeddings})

    with open (pool_data['outputPath'],"wb") as f:
        pickle.dump(output,f)


def main():
    image_paths = get_image_paths("lfw")

    # Define the number of processes to be used by the pool
    # Each process takes one core in the CPU
    PROCESSES = 6
    IMGS_PER_PROCESS = ceil(len(image_paths)/PROCESSES)

    # Split the images into equal sized batches for each process
    # Since we only need the embeddings for all the images the data can be split
    # into equal sized batches and each process can then independently compute the embeddings for the images.
    # The embeddings can then be concatenated after all of them are finished

    split_paths = []
    for i in range(0,len(image_paths),IMGS_PER_PROCESS):
        split_paths.append(image_paths[i:i+IMGS_PER_PROCESS])

    # Each process saves the embeddings computed by it into a pickle file in a temporary folder.
    # The temporary pickle files can then be loaded and we can generate a single pickle file containing all the embeddings for our data
    split_data = []
    for pool_id,batch in enumerate(split_paths):
        temp_path = os.path.join("temp","pool_{}.pickle".format(pool_id))

        pool_data = {
            "poolId":pool_id,
            "imagePaths": batch,
            "tempPath": temp_path
        }

        split_data.append(data)
    
    pool = multiprocessing.Pool(processes=PROCESSES)
    pool.map(compute_embedding,split_data)

    pool.close()
    pool.join()

    concat_embeddings = []

    for filename in os.listdir("temp"):
        data = pickle.load(open(os.path.join("temp",filename),"rb"))

        for dictionary in data:
            concat_embeddings.append(dictionary)
    
    with open("embeddings.pickle","wb") as f:
        f.write(pickle.dump(concat_embeddings,f))
    
    
            
    


    
    

if __name__ == "__main__":
    main()