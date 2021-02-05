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
    """Generates a list of the image paths in a root directory
       The directory must be structured like this
       |--root_dir
       |  |
       |  |--subfolder_1
       |  |  |
       |  |  |--image_1.jpg
       |  |  |--image2.jpg
       |  |  
       |  |--subfolder_2


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

    f = open(pool_data['outputPath'],"wb") #rename
    f.write(pickle.dumps(output))
    f.close()

def main():
    image_paths = get_image_paths("lfw")

    PROCESSES = 6
    IMGS_PER_PROCESS = ceil(len(image_paths)/PROCESSES)

    split_paths = []
    for i in range(0,len(image_paths),IMGS_PER_PROCESS):
        split_paths.append(image_paths[i:i+IMGS_PER_PROCESS])

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


    
    

if __name__ == "__main__":
    main()