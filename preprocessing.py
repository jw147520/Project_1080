import os
import numpy as np
import random
import pickle
import gzip
from deepgtav.messages import frame2numpy
import cv2

dataset_path = 'dataset.pz'


# 그냥 아래의 수치를 그대로 쓰면 되나??
def normalize(image):
    image[:, :, 0] -= 94.9449
    image[:, :, 0] /= 58.6121

    image[:, :, 1] -= 103.599
    image[:, :, 1] /= 61.6239

    image[:, :, 2] -= 92.9077
    image[:, :, 2] /= 68.66

    return image

def load_batches(verbose=1, samples_per_batch=1000):
    # Generator for loading batches of frames

    print("Loading dataset file...")
    dataset = gzip.open(dataset_path)
    print("Finished loading dataset file.")

    batch_count = 0
    abandon = 60  # 처음 버릴 frame 수
    abandoned = False
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

                    if batch_count == 0 and abandoned is False and count < abandon:  # abandon 만큼 첫 frame 은 버린다.
                        count += 1
                        if count == abandon:
                            count = 0
                            abandoned = True
                        continue


                    image = data_dct['frame']
                    # create_dataset.py 에서
                    # client.recvMessage_noSave() 를 통해서 message 를 받아오고
                    # client.save_to_datafile() 을 통해서 저장했다면 아래처럼
                    # frame2numpy() 를 해줄 필요가 없다.
                    # image = frame2numpy(frame, (800, 600))

                    # imshow 는 테스트 코드였음. 추후에 지워줄것!!
                    # cv2.imshow('imgae', image)
                    # cv2.waitKey(0)

                    # 아래 Simple preprocessing 은 그냥 down sizing 인듯 하다.
                    # image normalization 을 하면 필요 없을듯 하다. <중요>
                    image = np.float64(image)
                    image = normalize(image)  # Normalization
                    # image = ((image / 255) - .5) * 2  # Simple preprocessing
                    
                    # Train test split
                    # TODO: Dynamic train test split | Test series at end of batch
                    if (count % 5) != 0:  # Train
                        x_train.append(image)
                        # SantosNet 에서는 steering 을 1~1000 의 정수로 categorize 하였음.
                        # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
                        # y_train.append(int(float(data_dct['steering']) * 500) + 500)
                        y_train.append(data_dct['steering'])
                    else:  # Test
                        x_test.append(image)
                        # SantosNet 에서는 steering 을 1~1000 의 정수로 categorize 하였음.
                        # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
                        # y_test.append(int(float(data_dct['steering']) * 500) + 500)
                        y_test.append(data_dct['steering'])
                    
                    count += 1
                    if (count % 250) == 0 and verbose == 1:
                        print('     ' + str(count) + ' data points loaded in batch.')
            print('Batch loaded.')
            batch_count += 1
            yield x_train, y_train, x_test, y_test
        except EOFError:  # Breaks at end of file
            break
