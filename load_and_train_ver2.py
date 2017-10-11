import warnings
import numpy as np
import h5py
from preprocessing import load_batches
from keras.models import load_model
from keras.utils import np_utils
from model import *

input_shape = (800, 600, 3)  # input shape 을 이미지의 shape 에서 바로 읽어오도록 바꿀 필요가 있음
print("Loading Model ...")
model = get_model(input_shape=input_shape)
print("Model Loaded. Compiling...")
sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)  # SGD Optimizer 사용
model.compile(optimizer=sgd, loss="mse")  # loss function 은 논문을 따라 "mean squared error"
model.summary()

print("Starting Training...")
batch_count = 0
try:
    for i in range(0, 5):
        print('----------- On Epoch: ' + str(i) + ' ----------')
        for x_train, y_train, x_test, y_test in load_batches():
            # Model input requires numpy array
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            # Classification to one-hot vector
            y_train = np_utils.to_categorical(y_train, num_classes=1000)
            y_test = np_utils.to_categorical(y_test, num_classes=1000)
            # Fit model to batch
            model.fit(x_train, y_train, verbose=1, epochs=1, validation_data=(x_test, y_test))

            batch_count += 1
            # Save a checkpoint
            if (batch_count % 20) == 0:
                print('Saving checkpoint ' + str(batch_count))
                model.save('model_checkpoint' + batch_count + '.h5')
                print('Checkpoint saved. Continuing...')
except Exception as e:
    print('Excepted with ' + str(e))
    print('Saving model...')
    model.save('model_trained_categorical.h5')
    print('Model saved.')