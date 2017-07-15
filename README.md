# Image_Classification_CNN
The above project aims at harnessing the Keras library and its various methods to build a image 
classifier that takes input an image of dog or a cat and classifies it accordingly. This is the 
first step towards learing to use CNN using Keras and Tensorflow for doing image processing at 
great scale.

Below I will explain step by step procedure of building the classifier using Keras and CNN.

# Step1.
'''python
import all necessary libraries including keras and matplotlib
import os
from random import randint
from scipy import ndimage, misc
from skimage import io
from itertools import islice
import numpy as np
from PIL import Image,ImageOps
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten,Reshape
import matplotlib.pyplot as plt
'''
...
...
# step2.
Next step is to read the image from the files and store it in an array format using img_to_array function 
from numpy library.
reshape the image to improve the proccessing speed. (As taking the entire image size would increase the computation speed 
of the program.

# step3.

Initialize the classifier
/
'''python
classifier = Sequential()
'''

# step4.

Add multiple convolution and pooling layer to filter out the image and thus creating a feature map.
/
'''python
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
'''

# step5.
Flatten the imput matrix, so that it can be feeded as an input to the dense CNN layer.
classifier.add(Flatten())

# step6. 
Add a dense layer and then the output layer using   'relu' and 'sigmoid' activation function

# step7.
compile the model using .compile method and then finally fit the model using training and test data set

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)

Note to create a validation set for testing the performance of the model and avoid model overfitting.
# Sample training images (Input)
![cat 4001](https://user-images.githubusercontent.com/14236684/28104500-4fc95b1a-66a9-11e7-99a4-5f99131a3f7c.jpg)
![dog 4003](https://user-images.githubusercontent.com/14236684/28104580-cd03c11a-66a9-11e7-875f-71a425b946f8.jpg)
