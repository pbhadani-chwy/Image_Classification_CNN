# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN
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
import matplotlib.patches as patches
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

images = np.ndarray(shape=(8000,3,128,128),dtype=np.float32)

i = 0
label = []
for root, dirnames, filenames in os.walk("C:\\Users\\pbhadani\\Documents\\Deep Learning\\Deep-Learning-A-Z\\Deep Learning A-Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\Convolutional_Neural_Networks\\dataset\\training_set\\dogs"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        head,tail = os.path.split(filepath)  #split file into path + image name
        label.append(head.split('\\')[10])
        #print(label)
        image = load_img(filepath)
        image=image.resize((128,128),Image.LANCZOS)
        image=img_to_array(image)
        #print(image)
        images[i]=image.reshape(3,128,128)
        i += 1

print(i)
for i in range (3990,4010):
    fig,ax = plt.subplots(1)
    
    # Display the image
    index=randint(0,49)
    ax.imshow(images[i].reshape(128,128,3))
    plt.show()
    # Create a Rectangle patch
'''    x=rx[index]-lx[index]
    y=ry[index]-ly[index]
    rect1 = patches.Rectangle((lx[i],ly[i]),rx[i],ry[i],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect1)'''
    
   


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'