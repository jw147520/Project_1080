from keras.models import load_model
from model import *
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy
from deepgtav.client import Client
import win32gui
import cv2
import gzip
import numpy as np
from PIL import ImageGrab
from preprocessing import normalize
import pickle

model_path = 'model_checkpoint_numeric_ver2_280.h5'
print("Loading Model...")
model = load_model(model_path)
print("Model Loaded. Compiling...")
sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)  # SGD Optimizer 사용
model.compile(optimizer=sgd, loss="mse")  # loss function 은 논문을 따라 "mean squared error"
model.summary()

if input("Continue?") == "y": # Wait until you load GTA V to continue, else can't connect to DeepGTAV
    print("Conintuing...")

dataset_path = 'dataset.pz'
dataset = gzip.open(dataset_path)

count = 0
print("Starting Loop...")
while True:

    try:
        data_dct = pickle.load(dataset)
        frame_image = data_dct['frame']
        true_steering = data_dct['steering']
        frame_image = frame2numpy(frame_image, (800, 600))
        frame_image = cv2.flip(frame_image, 1)
        print(frame_image.shape)
        print(type(frame_image))
        # frame_image = ((frame_image / 255) - .5) * 2  # Simple preprocessing

        cv2.imshow('img', frame_image)
        cv2.waitKey(1)

        # Corrects for model input shape
        # frame_image = image.img_to_array(frame_image)
        frame_image = np.reshape(frame_image, (1,) + frame_image.shape)

        prediction = model.predict(frame_image)
        steering = prediction[0][0]
        steering = float(steering)
        print("true steering : " + str(true_steering) + "   predicted steering : " + str(steering))
        # Mutiplication scales decimal prediction for harder turning
        count += 1

    except Exception as e:
        print("Excepted as: " + str(e))
        continue

