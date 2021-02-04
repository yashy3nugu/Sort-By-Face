import multiprocessing
import cv2
import numpy as np
import os
import pickle
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

    for rootDir,directory,filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
                yield os.path.join(rootDir,filename)

def compute_embedding(split_data,detector="HOG"):
    """Function used by each processing pool to compute embeddings for part of a dataset


    Args:
        split_data : 
    """

    output = []

    for count,path in enumerate(split_data['input_paths']):
        image = load_and_align(path,detector)

        if image is None: # Case where no faces were found in image
            continue

        embeddings = embedder.embeddings(image)

        if embeddings.shape[0]>1:
            for embedding in embeddings:
                output.append({"path":path,"embedding":embedding})
        else:
            output.append({"path":path,"embedding":embeddings})

    f = open(data['output_paths'],"wb") #rename
    f.write(pickle.dumps(output))
    f.close()
