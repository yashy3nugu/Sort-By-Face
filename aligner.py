import imutils
import dlib
import cv2
import numpy as np
import os

# Helper functions to align faces
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

def load_and_align(filepath,detector="HOG"):
    """
    Loads an image from the given filepath. It then gives the resulting
    array containing an aligned image from a face detector

    Args:
        filepath : Relative filepath to an image
        detector_type: The dlib detector which is to be used
    """

    if detector == "HOG":
        # Histogram of Gradients predictor (faster)
        face_detector = dlib.get_frontal_face_detector()

    elif detector == "CNN":
        # CNN based predictor (accurate but slower)
        face_detector = dlib.cnn_face_detection_model.v1()
    else:
        raise ValueError("Detector needs to be either HOG or CNN based detector")

    shape_predictor = dlib.shape_predictor('Weights/shape_predictor_68_face_landmarks.dat')

    # Resize and align the face for facenet detector (facenet expects 160 by 160 images)
    face_aligner = FaceAligner(shape_predictor, desiredFaceHeight=160,desiredFaceWidth=160)

    input_image = cv2.imread(filepath)

    # convert images to grayscale for the detector
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    rectangles = face_detector(gray_img,2)


    for rectangle in rectangles:

        (x,y,height,width) = rect_to_bb(rectangle)
        aligned_face = face_aligner.align(input_image,gray_img,rectangle)

    # Todo - Add blur filtering and other methods for image processing

    return aligned_face
        

    