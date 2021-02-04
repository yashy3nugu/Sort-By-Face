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
