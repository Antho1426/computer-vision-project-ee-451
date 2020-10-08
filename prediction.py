from __future__ import print_function


import numpy as np

from skimage.transform import rotate
from skimage.filters import median

from CNN import predict_CNN, RotNet
from MLP import predict_mlp
from utils import principal_axes

#-----
import skimage
from skimage.filters import threshold_multiotsu


import skimage
from skimage.filters import median

from frames import frame_dict
from box_finding import get_ROI
from MLP import predict_mlp
from redress import principal_axes
from CNN import predict_CNN, RotNet


import os
import cv2
import sys
import keras
import pickle
import random
import math as m
import skimage.io
import numpy as np
from time import perf_counter
from keras.models import Model
import matplotlib.pyplot as plt
from keras.datasets import mnist
# /!\ (cf.: https://stackoverflow.com/questions/53135439/issue-with-add-method-in-tensorflow-attributeerror-module-tensorflow-python)
# Don't use:
#from keras.models import load_model
# But rather:
# ("There seems to be some weird compatibility issues between keras and tensorflow.keras")
from tensorflow.keras.models import load_model
from skimage.morphology import square
from skimage.filters import threshold_otsu
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard



# Import some code
from utils_RotNet import display_examples, RotNetDataGenerator, angle_error, binarize_images, rotate_RotNet
import correct_rotation
#-----




# Setting the current working directory automatically
import os
working_directory_path = os.getcwd()

# Setting current working directory
project_path = working_directory_path
os.chdir(project_path)






def digit_prediction(box, box_bin, image, model, mu, std, method='mlp', redress=False, redress_bis=False, net=RotNet()) :
    """
    Function that gives the digit prediction eiher with a MLP or a CNN classifier.
    
    Inputs : - box --> Image to classify (binarised gray scale image)
             - box_bin --> binarized box to find the redressing angle
             - image --> frame of the video
             - model --> String with the name of the classifier model
             - mu --> Dataset mean
             - std --> Dataset standard deviation
             - method --> "mlp" or "cnn" string for the classifier type
             - redress --> if True data redressed with axes of inertia (0 or 180°)
             - net --> CNN layers architecture
    
    Outputs : - prediction --> integer digit prediction
    """
    
    # Preprocessing
    if redress :
        box1 = box.copy()
        alpha, _ = principal_axes(box_bin)
        alpha = np.degrees(alpha)
        box1 = rotate(box1, -alpha, cval=1.0, preserve_range=True)
        box1 = median(box1)
        
    if redress_bis :
        box2=box.copy()
        alpha, _ = principal_axes(box_bin)
        alpha = np.degrees(alpha)
        box2 = rotate(box2, -alpha + 180, cval=1.0, preserve_range=True)
        box2 = median(box2)
        
    # MPL model
    if method == 'mlp' :
        if not redress :
            box1 = box.copy()
            
        prediction1, prob1, _ = predict_mlp(box1, image, model, mu, std)
        
        if redress_bis :
            prediction2, prob2, _ = predict_mlp(box2, image, model, mu, std)
            
            if prob1[0][int(prediction1)] > prob2[0][int(prediction2)] :
                prediction = prediction1
                prob = prob1
            else :
                prediction = prediction2
                prob = prob2
        
        else :
            prediction = prediction1
            prob = prob1

    # CNN model
    elif method == 'CNN' :
        prediction = predict_CNN(box, model, net, mu, std)
        prob = -1
    
    if prediction == 9 :
        prediction = str(6)
    
    return prediction     #, prob


def digit_prediction_with_keras(box, box_bin, gray):
    _, _, rotated_img = predict_mlp(box, gray, 'mlp12', 1, 1)
    rotated_img_for_plot = rotated_img.copy()

    # Adding a dimension for consistence
    rotated_img = np.expand_dims(rotated_img, axis=0)
    rotated_img = np.expand_dims(rotated_img, axis=-1)

    # 6) Binarizing the test image
    rotated_img_bin = binarize_images(rotated_img)
    # rotated_img_bin_for_plt = np.squeeze(rotated_img_bin, axis=0); rotated_img_bin_for_plt = np.squeeze(rotated_img_bin_for_plt, axis=-1); plt.figure(figsize=(3,3)); plt.imshow(rotated_img_bin_for_plt)


    # 1) Loading the model
    model_location = os.path.join(project_path, 'rotnet_mnist.hdf5')
    model = load_model(model_location, custom_objects={'angle_error': angle_error})

    # 7) Predicting rotation angle
    #======================================
    output = model.predict(rotated_img_bin)
    #======================================
    predicted_angle = np.argmax(output)
    #print('Predicted angle: ', predicted_angle)

    # 8) Rotating back the rotated test image
    rotated_img_2D = rotated_img_for_plot.copy()
    corrected = rotate_RotNet(rotated_img_2D, -predicted_angle)

    # /!\ Cropping a bit the image in case the dimensions are bigger
    dim_cor = corrected.shape[0]
    dim_orig = 28
    if dim_cor > dim_orig:
        cropped_dist = m.ceil((dim_cor - dim_orig)/2)
        corrected = corrected[cropped_dist:dim_cor-cropped_dist, cropped_dist:dim_cor-cropped_dist]

    # Resizing corrected to make sure it has as expected a shape of dimension
    # (28, 28) (and NOT (30, 30) or whatever!!!)
    corrected = cv2.resize(corrected, (28, 28), interpolation = cv2.INTER_AREA)

    #mu, std = X_train.mean(), X_train.std()
    mu = 33.318447
    std = 78.567444

    # Standardizing "corrected"
    #print("corrected.shape: ", corrected.shape)
    corrected_linear = corrected.reshape(-1)
    #print("corrected_linear.shape: ", corrected_linear.shape)
    corrected_standardized = (corrected_linear - mu)/std
    # Adding a dimension for consistence (to have a 2D instead of a 1D array)
    corrected_standardized = np.expand_dims(corrected_standardized, axis=0)



    # Load the model from disk (trained on usual MNIST)
    #---------------------------------------
    #filename = 'mlp_model.sav'
    filename = 'mlp_model_rotated_mnist.sav'
    #---------------------------------------
    mlp = pickle.load(open(filename, 'rb'))

    # Predicting the class of the corrected image
    #====================================================
    # Predicting on the (somehow) re-oriented digit based on the CNN
    predicted_class = mlp.predict(corrected_standardized)
    #====================================================


    prediction = str(predicted_class[0])


    # Plotting the number and printing its prediction in the title of the figure
    #------
    # plt.figure(figsize=(3,3))
    # plt.imshow(corrected, cmap='gray')
    # plt.title(('corrected, pred: {}'.format(predicted_class)))
    # plt.show()
    # ------


    return prediction, predicted_angle