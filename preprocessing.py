import os
import numpy as np
import random
import pickle
import gzip
from deepgtav.messages import frame2numpy
import cv2

dataset_path = 'dataset.pz'


def load_batches(verbose=1, samples_per_batch=1000):
    # Generator for loading batches of frames
    print("fuck you")
    dataset = gzip.open(dataset_path)
    batch_count = 0
    abandon = 60  # 처음 버릴 frame 수
    # 시작시 끊김 현상, 상/하단부 메시지 등 ... 때문에

    while True:
        try:
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            count = 0

            print('----------- On Batch: ' + str(batch_count) + ' -----------')
            while count < samples_per_batch:
                    data_dct = pickle.load(dataset)  # 참고: pickle.load() 는 파일에서 한 줄씩 읽어온다.

                    if count < abandon:  # abandon 만큼 첫 frame 은 버린다.
                        count += 1
                        continue

                    image = data_dct['frame']
                    # create_dataset.py 에서
                    # client.recvMessage_noSave() 를 통해서 message 를 받아오고
                    # client.save_to_datafile() 을 통해서 저장했다면 아래처럼
                    # frame2numpy() 를 해줄 필요가 없다.
                    # image = frame2numpy(frame, (800, 600))

                    cv2.imshow('imgae', image)
                    cv2.waitKey(0)
                    image = ((image / 255) - .5) * 2  # Simple preprocessing
                    
                    # Train test split
                    # TODO: Dynamic train test split | Test series at end of batch
                    if (count % 5) != 0:  # Train
                        x_train.append(image)
                        # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
                        y_train.append(int(float(data_dct['steering']) * 500) + 500) 
                    else: # Test
                        x_test.append(image)
                        # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
                        y_test.append(int(float(data_dct['steering']) * 500) + 500)
                    
                    count += 1
                    if (count % 250) == 0 and verbose == 1:
                        print('     ' + str(count) + ' data points loaded in batch.')
            print('Batch loaded.')
            batch_count += 1
            yield x_train, y_train, x_test, y_test
        except EOFError:  # Breaks at end of file
            break
