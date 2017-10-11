import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.models import load_model

from keras.layers import Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers import Conv2D

from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint


# CNN architector of " End-to-End Learning for Self-Driving Cars"
# https://github.com/marshq/europilot
def get_model(input_shape):
    # input_shape : input image shape, (800, 600, 3)이 될 듯 하다.
    model = Sequential([
        Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape),
        BatchNormalization(axis=1),
        Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        BatchNormalization(axis=1),
        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dense(50, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='relu'),
        BatchNormalization(),
        Dense(1)
    ])

    # 이후에 본 코드에서 아래와 같이 model 을 정의한 후 compile 해주어야 함 - keras
    """
    model = get_model(input_shape)
    sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="mse")
    model.summary()
    """
    return model


