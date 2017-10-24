import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model

from keras.layers import *

from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint

from keras.applications.inception_v3 import *

# add new last layer, specifically fully-connected layer at the top of the network
def add_new_last_layer(base_model, nb_classes):
    """
    Add last FC layer at the top of the network
    :param base_model: keras model excluding top
    :param nb_classes: number of classes
    :return: new keras model with top layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    return model


# get InceptionV3 model
def get_inception(input_shape, nb_class):

    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1024, activation='relu')(x)
    prediction = Dense(nb_class, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)

    for layer in base_model.layers:
        layer.trainable = False

    return model


# CNN architecture of " End-to-End Learning for Self-Driving Cars"
def get_nvidia_model(input_shape):
    model = Sequential([
        Lambda(BatchNormalization(epsilon=0.001, axis=1, input_shape=input_shape)),
        Conv2D(24, kernel_size=(5, 5), border_mode='valid', strides=(2, 2), activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        Dropout(0.2),
        Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        Dropout(0.2),
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        Dropout(0.2),
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        Dropout(0.2),

        Flatten(),

        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='relu'),

        Dense(1)
    ])

    model.summary()
    return model


