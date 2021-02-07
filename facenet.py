import numpy as np
import tensorflow as tf
import cv2

def standardize(image):
    mean = np.mean(image, axis=(1,2,3),keepdims=True)
    std_dev = np.std(image,axis=(1,2,3),keepdims=True)
    std_dev = np.maximum(std_dev,1.0/np.sqrt(image.size))

    standardized_image = (image - mean)/std_dev
    return standardized_image