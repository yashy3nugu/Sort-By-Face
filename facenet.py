import numpy as np
import tensorflow as tf
import cv2
from aligner import load_and_align

def standardize(image):
    """converts the pixel values range to the one suitable for facenet

    Args:
        image : numpy array of an rgb image with pixels in range [0,255]

    Returns:
        standardized_image: image standardized for the facenet model
    """
    mean = np.mean(image, axis=(1,2,3),keepdims=True)
    std_dev = np.std(image,axis=(1,2,3),keepdims=True)
    std_dev = np.maximum(std_dev,1.0/np.sqrt(image.size))

    standardized_image = (image - mean)/std_dev
    return standardized_image

def normalize_emb(emb, axis=-1,eps=1e-10):
    """L2 normalizes the embeddings from the model

    Args:
        emb : numpy array of shape (1,512) containing the embedding from the model
        axis : axis on which to compute L2 norm
        eps : epsilon value to prevent division by zero
    """
    normalized_emb = emb / np.sqrt(np.maximum(np.sum(np.square(emb), axis=axis, keepdims=True), eps))
    return normalized_emb

def compute_embedding(img_path,model):

    # can be a single image or a batch of images depending on number of faces detected in the image
    images = load_and_align(img_path)
    # standardize them
    
    if images is None:
        return None
    images = standardize(images)

    # (batch_size,1,512)
    embeddings = model.predict(images)
    embeddings = normalize_emb(embeddings)

    return embeddings
