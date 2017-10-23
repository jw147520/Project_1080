# -*- coding: utf-8 -*- #
"""
Train Inception V3 model for Keras.
"""
import warnings
import numpy as np
import h5py
from preprocessing import load_batches_category
from keras.models import load_model
from keras.utils import np_utils
from model import *

input_shape = (600, 800, 3)  # input shape 을 이미지의 shape 에서 바로 읽어오도록 바꿀 필요가 있음
print("Loading Model ...")
model = get_inception(input_shape=input_shape, nb_class=1001)
# model = load_model('model_checkpoint.h5')  # 학습을 진행한 모델이 존재한다면 이 줄을 사용.
print("Model Loaded. Compiling...")
sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)  # SGD Optimizer 사용

model.compile(optimizer='rmsprop', loss="categorical_crossentropy")  # loss function 은 논문을 따라 "mean squared error"
model.summary()  # model summary 출력

# loss 기록을 위한 txt 파일
f = open("loss_inception.txt", 'w')

print("Starting Training...")
batch_count = 0
try:
    for i in range(0, 1):  # 데이터셋이 크면 한 번의 epoch 밖에 못돌릴 듯 하다.
        print('----------- On Epoch: ' + str(i) + ' ----------')
        for x_train, y_train, x_test, y_test in load_batches_category():
            # Model input requires numpy array
            x_train = np.array(x_train)
            x_test = np.array(x_test)

            # Inception V3 의 기본 구조인 classes=1000 (ImageNet data 의 특성 때문) 을 바꾸지 않고
            # 학습하기 위해 numeric value 를 categorical 로 바꿔준다.
            # SantosNet 참고.
            # Classification to one-hot vector
            print(y_train)
            y_train = np_utils.to_categorical(y_train, num_classes=1001)
            print(y_train)
            print('')
            print(y_test)
            y_test = np_utils.to_categorical(y_test, num_classes=1001)
            print(y_test)

            print("Finished making it to categorical")

            # Fit model to batch
            train_history = model.fit(x_train, y_train, verbose=1, epochs=1, validation_data=(x_test, y_test))

            # batch_count trainint_loss valication_loss 의 형태로 기록
            f.write(str(batch_count) + ' ' + str(train_history.history['loss']) + ' ' + str(train_history.history['val_loss']))

            batch_count += 1
            # Save a checkpoint
            if (batch_count % 100) == 0:
                print('Saving checkpoint ' + str(batch_count))
                model.save('v3_model_checkpoint' + str(batch_count) + '.h5')
                print('Checkpoint saved. Continuing...')

except Exception as e:
    print('Excepted with ' + str(e))
    print('Saving model...')
    model.save('model_trained_v3.h5')
    print('Model saved.')