import warnings
import numpy as np
import h5py
from preprocessing import load_batches_numeric
from keras.models import load_model
from keras.utils import np_utils
from model import *

input_shape = (600, 800, 3)  # frame2numpy 를 거친 input 은 [800, 600] 이 아닌 [600, 800] 이다.
print("Loading Model ...")
model = get_nvidia_model(input_shape=input_shape)
# model = load_model('model_checkpoint.h5')  # 학습을 진행한 모델이 존재한다면 이 줄을 사용.
print("Model Loaded. Compiling...")
sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)  # SGD Optimizer 사용
model.compile(optimizer='adam', loss="mse")  # loss function 은 논문을 따라 "mean squared error"
model.summary()

# loss 기록을 위한 txt 파일
f = open("loss_nvidia.txt", 'w')

print("Starting Training...")
batch_count = 0
try:
    for i in range(0, 1):
        print('----------- On Epoch: ' + str(i) + ' ----------')
        for x_train, y_train, x_test, y_test in load_batches_numeric():
            # Model input requires numpy array
            x_train = np.array(x_train)
            x_test = np.array(x_test)

            # Fit model to batch
            train_history = model.fit(x_train, y_train, verbose=1, epochs=1, validation_data=(x_test, y_test))

            # batch_count training_loss validation_loss 의 형태로 기록
            f.write(str(batch_count) + ' ' + str(train_history.history['loss'][0]) + ' ' + str(
                train_history.history['val_loss'][0]))

            batch_count += 1
            # Save a checkpoint
            if (batch_count % 20) == 0:
                print('Saving checkpoint ' + str(batch_count))
                model.save('model_checkpoint_numeric_' + str(batch_count) + '.h5')
                print('Checkpoint saved. Continuing...')
except Exception as e:
    print('Excepted with ' + str(e))
    print('Saving model...')
    model.save('excepted_model_numeric.h5')
    print('Model saved.')

# save final model
model.save('model_numeric.h5')