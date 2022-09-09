# This neural network will identify whether an image contains a ship or not
# Based on CNN built in Udemy course by Eremenko and de Ponteves
# Dataset taken from Kaggle: Ships in Satellite Imagery (San Francisco Bay)
# URL on 06/02/2020: https://www.kaggle.com/rhammell/ships-in-satellite-imagery

##########################################
# Step 1: Import necessary Keras modules #
##########################################

# keras.models Sequential defines the model as a sequence of layers,
#   ie from the input layer to the convolution layer to the pooling layer...
#   in a sequence
from keras.models import Sequential

# keras.layers Convolution2D is the convolution step, and in 2d as images are...
#   in 2 dimensions. This package will represent each convolutional layer
from keras.layers import Convolution2D

# keras.layers MaxPooling2D is the pooling step in 2 dimensions, used to...
#   continue on to the pooling step from the convolution step
from keras.layers import MaxPooling2D

# keras.layers Flatten used to turn all the pooled feature maps into one...
#   large feature vector
from keras.layers import Flatten

# keras.layers Dense used to create a fully connected layer just like a...
#   classic ANN
from keras.layers import Dense

# The following 2 lines might be needed if running on Mac
    # uncomment below if duplicate library error occurs
#import os
#s.environ['KMP_DUPLICATE_LIB_OK']='True'

#############################
# Step 2: Convolution layer #
#############################
# Initialise the Convolutional Neural Network
#   called classifier as we are classifying images, as a Sequential object
classifier = Sequential()

# Add a Convolution2D layer to the NN
classifier.add(
    Convolution2D(
        # :filters: The number of filters we want to create
            # 32 is common practice to start with, then add more conv layers
        filters = 32, # so 32 filters -> 32 feature maps

        # :kernel_size: the size of the filter/feature detector/kernel
        kernel_size = (3,3),

        # :input_shape: The shape of each image;
            # For Theano backend:
                # 3 channels for RGB or 1 for B&W,
                # number of rows of pixels, number of columns of pixels
                # e.g. input_shape = (3, 80, 80)
            # For TensorFlow backend:
                # number of rows of pixels, number of columns of pixels
                # 3 channels for RGB or 1 for B&W,
                # e.g. input_shape = (80, 80, 3)
        input_shape = (80, 80, 3), # colour images all 80x80 as specified by...
                                    # the dataset, using TensorFlow backend

        # :activation: Enter str of the activation function desired to ensure
            # non-linearity, in case of negative values from conv operation
        activation = "relu"
    )
)

#######################################
# Step 3: Pooling & Flattening Layers #
#######################################
# Add a pooling layer to the CNN
classifier.add(
    MaxPooling2D(
        # :pool_size: The size of the pooling filter applied to the feature map
            # 2x2 is very common for a pooling filter
        pool_size = (2, 2)
        # :strides: if not input, will default to pool_size
    )
)

# Add a flattening layer to the CNN
classifier.add(Flatten())

##################################
# Step 4: Fully Connected Layers #
##################################
# Add the first fully connected layer
classifier.add(
    Dense(
        # :units: The number of nodes in the Dense layer
            # Common practice to pick power of 2
        units = 128,

        # :activation: The desired activation function applied to the nodes
        activation = 'relu'
    )
)

# Add second and final fully connected layer for the output
classifier.add(
    Dense(
        # :units: Now we will only have 1 output node, as the...
            # ... classification result is binary
        units = 1,

        # :activation: The sigmoid function will be used to return a value...
            # ... between 0 and 1, probability of ship existing
        activation = 'sigmoid'
    )
)

###########################
# Step 5: Compile the CNN #
###########################

classifier.compile(
    # :optimizer: Adjustment of the weights based on the Stochastic Gradient...
        # ... Descent algorithm, of which 'Adam' is an efficient version
        # 'Adam'/SGD is dependent on a loss function, which is defined next...
    optimizer = 'adam',

    # :loss: The loss function that we are looking to minimise
        # In this case, we use binary cross entropy
            # Binary because of the binary desired output
            # Cross entropy because of the sigmoid activation function used
    loss = 'binary_crossentropy',

    # :metrics: Used to evaluate the performance of the model based on parameter
        # 'accuracy' compares true to predicted values
    metrics = ['accuracy']
)

##############################
# Step 6: Image Augmentation #
##############################
# Image augmentation prevents overfitting to the training set, which would...
    # ... result in high accuracy on the training set but lower accuracy on...
    # ... the test set
# Overfitting can occur when there are only a few training values
# Augmentation applies random transformations to batches within the dataset...
    # ... resulting in theoretically more images to train with
    # Enriching the dataset without adding any more images

from keras.preprocessing.image import ImageDataGenerator

# :ImageDataGenerator: intercepts the images being fed to the NN, applies...
    # ... random transformations to the images, and feeds ONLY the...
    # ... transformed images to the NN.
    # As a result, the NN is only ever seeing new images each epoch, as the...
    # ... images it is being fed have been transformed differently than the last
train_datagen = ImageDataGenerator(
    # :rescale: Alters the values of each pixel by the scale factor
        # In this case, each pixel will contain a value from 0 -> 1 as max...
        # ... pixel value is 255
    rescale = 1./255,

    # :shear_range: Shear angle in counterclockwise direction in degrees
        # Shear transformation of the images
    shear_range = 0.2,

    # :zoom_range: If float, sets [lower, upper] zoom values to...
        # [1-zoom_range, 1+zoom_range]. Or just input [lower, upper]
    zoom_range = 0.2,

    # :horizontal_flip: Randomly flip inputs horizontally
    horizontal_flip = True
)

# In the testing/validation dataset, we will just rescale the images so pixels
    # have a value between 0 - 1
valid_datagen = ImageDataGenerator(rescale = 1./255)

###############################
# Step 7: Apply preprocessing #
###############################

# Define the training set as flow_from_directory
training_set = train_datagen.flow_from_directory(
    # Path to the training dataset
    'ships-in-satellite-imagery/training_set',

    # :target_size: size of the input images
    target_size = (80,80),

    # :batch_size: size of batches of random samples of images to be fed to NN,
        # after which the weights will be updated
    batch_size = 32,

    # :class_mode: 'binary' if 2 classes, else 'categorical'
    class_mode = 'binary'
)

# Define the test set as flow_from_directory
valid_set = valid_datagen.flow_from_directory(
    # Path to the test dataset
    'ships-in-satellite-imagery/valid_set',

    # :target_size: size of the input images
    target_size = (80,80),

    # :batch_size: size of batches of random samples of images to be fed to NN,
        # after which the weights will be updated
    batch_size = 32,

    # :class_mode: 'binary' if 2 classes, else 'categorical'
    class_mode = 'binary'
)

###########################################
# Step 8: Apply NN to Preprocessed Images #
###########################################
classifier.fit_generator(
    # :generator: The generated dataset used to train the classifier NN
    generator = training_set,

    # :steps_per_epoch: Number of batches to run through per epoch
        # 2000 no ship, 677 ship = 2677 samples in training set
        # 2677/32 = 83.66 -> 84 batches
    steps_per_epoch = 84,

    # :epochs: number of epochs
    epochs = 25,

    # :validation_data: The dataset with the validation data
    validation_data = valid_set,

    # :validation_steps: number of batches in the validation dataset
        # 1000 no ship, 323 ship = 1323 samples
        # 1323/32 = 41.34 ~ 42 batches
    validation_steps = 42
)
