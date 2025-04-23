import argparse
import cv2
from os.path import dirname, join
import numpy as np

import random

def add_salt_and_pepper_noise(image, salt_prob=0.2, pepper_prob=0.1):
    noise = np.zeros_like(image)
    salt = np.random.rand(*image.shape) < salt_prob
    pepper = np.random.rand(*image.shape) < pepper_prob
    noise[salt] = 255
    noise[pepper] = 0
    noisy_image = cv2.add(image, noise)
    return noisy_image

def print_image(image):
    if image is not None:
        # Display the image in a window
        cv2.imshow("Image", image)
        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def load_caffe_models():
    """
    This function loads caffe model.

    .prototxt file holds overall information about the neural network, such as:
    list of layers, connections between layers,parameters of each layers, and input/output dimensions.
    .caffemodel file stores weights of layers of Neural network
    """
    protoPath = join(dirname(__file__), "deploy_gender.prototxt")
    modelPath = join(dirname(__file__), "gender_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
        protoPath, modelPath
    )
    return gender_net


def classify(imagePaths):
    target = []
    label = []
    i = 0
    for imagePath in imagePaths:
        gender_net = load_caffe_models()

        gender_list = ["0", "1"]
        """
        Preprocess image and prepare it for the classification.

        scalefactor: after mean subtraction image can be scaled by the scalefactor
        size: size that Convolutional Neural Network expects.
        mean values here are for mean subtraction, usually the mean values that were used for
        training are used for classification as well
        """
        image = cv2.imread(imagePath)
        image = add_salt_and_pepper_noise(image)
        mean = (104, 117, 123)
        blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), mean, swapRB=False)

        target.append(imagePath.split("_")[3])
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        label.append(gender)
    return (target, label)
