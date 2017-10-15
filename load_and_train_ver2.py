import warnings
import numpy as np
import h5py
from preprocessing import load_batches
from keras.models import load_model
from keras.utils import np_utils
from model import *

input_shape = (600, 800, 3)  # input shape 을 이미지의 shape 에서 바로 읽어오도록 바꿀 필요가 있음
print("Loading Model ...")
model = get_model(input_shape=input_shape)
# model = load_model('model_checkpoint.h5')  # 학습을 진행한 모델이 존재한다면 이 줄을 사용.
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

            # SantosNet 에서는 steering 을 1~1000 으로 categorize 하여 사용하였으나 이렇게 하지 않을거임.
            # Classification to one-hot vector
            # y_train = np_utils.to_categorical(y_train, num_classes=1000)
            # y_test = np_utils.to_categorical(y_test, num_classes=1000)

            # Fit model to batch
            # for x1, y1, x2, y2 in x_train, y_train, x_test, y_test:
            #    model.fit(x1, y1, verbose=1, epochs=1, validation_data=(x2, y2))
            model.fit(x_train, y_train, verbose=1, epochs=1, validation_data=(x_test, y_test))

            batch_count += 1
            # Save a checkpoint
            if (batch_count % 20) == 0:
                print('Saving checkpoint ' + str(batch_count))
                model.save('model_checkpoint_3_' + str(batch_count) + '.h5')
                print('Checkpoint saved. Continuing...')
except Exception as e:
    print('Excepted with ' + str(e))
    print('Saving model...')
    model.save('model_trained.h5')
    print('Model saved.')