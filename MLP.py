import os
import gzip

import numpy as np
import random
import matplotlib.pyplot as plt

from skimage.transform import rotate, resize
from skimage.filters import median, threshold_multiotsu
from skimage.util import pad, invert

from joblib import dump, load

from sklearn.neural_network import MLPClassifier

from utils import principal_axes


def extract_data(filename, image_shape, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data


def extract_labels(filename, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def MNIST_dataloader(data_base_path, data_folder) :
    image_shape = (28, 28)
    train_set_size = 60000
    test_set_size = 10000
    
    data_part2_folder = os.path.join(data_base_path, data_folder, 'part2')
    
    train_images_path = os.path.join(data_part2_folder, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_part2_folder, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_part2_folder, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_part2_folder, 't10k-labels-idx1-ubyte.gz')
    
    train_images = extract_data(train_images_path, image_shape, train_set_size)
    test_images = extract_data(test_images_path, image_shape, test_set_size)
    train_labels = extract_labels(train_labels_path, train_set_size)
    test_labels = extract_labels(test_labels_path, test_set_size)
    
    """
    # Training set binarisation
    for i, im in enumerate(train_images) :
        im[np.where(im>120)] = 255
        im[np.where(im<=120)] = 0
        
        train_images[i] = im
     
    # Testing set binarisation 
    for i, im in enumerate(test_images) :
        im[np.where(im>120)] = 255
        im[np.where(im<=120)] = 0
        
        test_images[i] = im
    """
    return train_images, train_labels, test_images, test_labels


def rotate_dataset(train_images, test_images) :
    for i, im in enumerate(train_images) :
        im = rotate(im, random.randint(0,360))#, preserve_range=True)
        #im[np.where(im>120)] = 255
        #im[np.where(im<=120)] = 0
        
        train_images[i] = im
        
    for i, im in enumerate(test_images) :
        im = rotate(im, random.randint(0,360))#, preserve_range=True)
        #im[np.where(im>120)] = 255
        #im[np.where(im<=120)] = 0
        
        test_images[i] = im

    return train_images, test_images

def redress_data(train_images, test_images) :
    for i, im in enumerate(train_images) :
        alpha, _ = principal_axes(im)
        alpha = np.degrees(alpha)
        im = rotate(im, -alpha)#, preserve_range=True)
        #im = median(im)
        
        im[np.where(im>120)] = 255
        im[np.where(im<=120)] = 0
        
        train_images[i] = im
        
    
    for i, im in enumerate(test_images) :
        alpha, _ = principal_axes(im)
        alpha = np.degrees(alpha)
        im = rotate(im, -alpha)#, preserve_range=True)
        #im = median(im)
        
        im[np.where(im>120)] = 255
        im[np.where(im<=120)] = 0
        
        test_images[i] = im
        
    
    return train_images, test_images

def remove_6_9(train_images, train_labels, test_images, test_labels) :
    train = list()
    test = list()
    for i in range(train_images.shape[0]) :
        if not (train_labels[i] == 6) and not (train_labels[i] == 9) :
            train.append(i)
            #train_images = np.delete(train_images, i)
            #train_labels = np.delete(train_labels, i)
            #print(i)
            
        if i < test_images.shape[0] :
            if not (test_labels[i] == 6) and not (test_labels[i] == 9) :
                test.append(i)
                #test_images = np.delete(test_images, i)
                #test_labels = np.delete(test_labels, i)
    
    train_im = np.zeros([len(train), train_images.shape[1], train_images.shape[2]])
    train_lab = np.zeros([len(train)])
    for j, ind in enumerate(train) :
      train_im[j] = train_images[ind]
      train_lab[j] = train_labels[ind]
      
    test_im = np.zeros([len(test), test_images.shape[1], test_images.shape[2]])
    test_lab = np.zeros([len(test)]) 
    for j, ind in enumerate(test) :
        test_im[j] = test_images[ind]
        test_lab[j] = test_labels[ind]
    
    return train_im, train_lab, test_im, test_lab

def compute_error(ground_truth, prediction):
    errors_location = np.where((ground_truth - prediction) != 0)
    nb_errors = len(errors_location[0])
    error_percentage = round((nb_errors / ground_truth.shape[0]) * 100, 3)
    model_accuracy = error_percentage
    return model_accuracy


def compute_error_by_class(ground_truth, prediction) :
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #classes = [0,1, 2, 3, 4, 5, 7, 8]
    model_accuracy = np.zeros(len(classes), dtype=float)
    i = 0
    for cl in classes:
        locations = np.where(ground_truth == cl)
        errors_location = np.where((ground_truth[locations] - prediction[locations]) != 0)
        nb_errors = len(errors_location[0])
        error_percentage = round((nb_errors/ ground_truth.shape[0])*100, 3)
        model_accuracy[i] = error_percentage
        i += 1
    return model_accuracy


def training(train_images, train_labels, model_name,
              hidden_layer_sizes=(100, ), activation='relu',
              solver='adam', alpha=0.0001, batch_size='auto',
              learning_rate='constant', learning_rate_init=0.001,
              power_t=0.5, max_iter=200, shuffle=True,
              random_state=None, tol=0.0001, verbose=False,
              warm_start=False, momentum=0.9, nesterovs_momentum=True,
              early_stopping=False, validation_fraction=0.1,
              beta_1=0.9, beta_2=0.999, epsilon=1e-08,
              n_iter_no_change=10) :
    
    # Data flatenning
        # Going from dimensions (60000, 28, 28) to (60000, 784)
    X_train = train_images.reshape(train_images.shape[0],-1)
    
    # Standardization
    mu, std = X_train.mean(), X_train.std()
    # mu = 33.318447, std = 78.567444
    print(mu, std)
    
    X_train_standardized = (X_train - mu)/std
    
    y_train = train_labels
    
    # Classifier
    mlp = MLPClassifier(hidden_layer_sizes, activation,
                        solver, alpha, batch_size,
                        learning_rate, learning_rate_init,
                        power_t, max_iter, shuffle,
                        random_state, tol, verbose,
                        warm_start, momentum, nesterovs_momentum,
                        early_stopping, validation_fraction,
                        beta_1, beta_2, epsilon, n_iter_no_change)
    
    mlp.fit(X_train_standardized, y_train)
    dump(mlp, model_name)
    
    return mlp, mu, std


def testing(model_name, test_images, test_labels, mu, std) :
    model = load(model_name)
    X_test = test_images.reshape(test_images.shape[0], -1)
    X_test_standardized = (X_test - mu)/std
    y_test = test_labels
    y_predicted = model.predict(X_test_standardized)
    
    # Error
    print("Model accuracy at testing [%]: ", 100 - compute_error(y_test, y_predicted))
    model_accuracy_at_testing = compute_error_by_class(y_test, y_predicted)
    print("Class error rates at testing [%]:", model_accuracy_at_testing)
    
    return model_accuracy_at_testing


def predict_mlp(image, im_origin, model_name, mu, std) :
    """
    Function that to the prediction based on a mlp model
    
    Inputs : - Image --> original image to pred
             - model_name --> name of the model (in the same folder as the main)
    """
    
    # Loading the model
    model = load(model_name)
    
    # Find maximal dimension
    dim = max(image.shape)
    ind = np.argmin(image.shape)
    
    # Padding to have sqaure image
    if ind == 0 :
        i = 0
        while image.shape[ind] != dim :
            if i%2 == 0 :
                image = pad(image, ((1,0),(0,0)), mode='maximum')
            else :
                image = pad(image, ((0,1),(0,0)), mode='maximum')
            i+=1
            
    else :
        i = 0
        while image.shape[ind] != dim :
            if i%2 == 0 :
                image = pad(image, ((0,0),(1,0)), mode='maximum')
            else :
                image = pad(image, ((0,0),(0,1)), mode='maximum')
            i+=1
        
    # Resize to have a correct dimension for the input of the mlp
    image = resize(image, (28, 28))
    image = median(image)
    
    # Threshold for the inarisation
    thresholds = threshold_multiotsu(im_origin, classes=2)
    thresh_background = thresholds[0]
    
    # Binarisation
    im = image.copy()
    """
    im[np.where(image>=thresh_background)] = 0
    im[np.where(image<thresh_background)] = 255
    """
    im[np.where(image>=thresh_background)] = 1
    im = invert(im)
    m = np.max(im)
    im = im/m*255

    test = im.copy()
    
    # Copy for the plot
    #plot = im.copy()
    
    #mu = image.mean()
    #std = image.std()
    im = im.reshape(1, -1)
    
    # Normalisation
    im_norm = (im - mu)/std
    prediction = model.predict(im_norm)
    prediction_string = str(prediction[0])
    prob = model.predict_proba(im_norm)
    
    # Plot
    #plt.imshow(plot, cmap='gray')
    #plt.title('Rotated binary box : %d , %f' %(int(prediction), prob[0][int(prediction)]))
    #plt.show()
    
    return prediction_string, prob, test