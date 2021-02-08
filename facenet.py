import imutils
import dlib
import os
import cv2
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

def load_and_align(filepath):
    """
    Loads an image from the given filepath. It then gives the resulting
    array containing an aligned image from a face detector

    Args:
        filepath : Relative filepath to an image
        detector_type: The dlib detector which is to be used
    """

    face_detector = dlib.get_frontal_face_detector()

    shape_predictor = dlib.shape_predictor(
        'Weights/shape_predictor_68_face_landmarks.dat')

    # Resize and align the face for facenet detector (facenet expects 160 by 160 images)
    face_aligner = FaceAligner(
        shape_predictor, desiredFaceHeight=160, desiredFaceWidth=160)

    input_image = cv2.imread(filepath)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    height, width, _ = input_image.shape

    # Discard any low resolution images
    if height < 160 or width < 160:
        return None

    # convert images to grayscale for the detector
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    rectangles = face_detector(gray_img, 2)

    if len(rectangles) == 1:
        # (x,y,height,width) = rect_to_bb(rectangle)
        aligned_face = face_aligner.align(input_image, gray_img, rectangles[0])

        return np.expand_dims(aligned_face, axis=0)

    # If multiple faces are found then return a batch of images
    elif len(rectangles) > 0:

        aligned_faces = []
        for rectangle in rectangles:

            # (x,y,height,width) = rect_to_bb(rectangle)
            aligned_face = face_aligner.align(input_image, gray_img, rectangle)
            aligned_faces.append(aligned_face)

        return np.array(aligned_faces)

        # Todo - Add blur filtering and other methods for image processing

    else:
        return None

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
    """Computes the embedding(s) for the face(s) in the image at the given path

        NOTE: The model is not loaded in this function to prevent reading the *.h5 file
        to load the model everytime this function is called

    Args:
        img_path : relative path to the image
        model : keras model to compute embeddings

    Returns:
        embeddings: numpy array of shape (number of detected faces,dimension of embedding) containing the embeddings for the detected faces
    """
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
