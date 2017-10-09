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
    while True:
        try:
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            count = 0
            print('----------- On Batch: ' + str(batch_count) + ' -----------')
            while count < samples_per_batch:
                    data_dct = pickle.load(dataset)
                    frame = data_dct['frame']
                    image = frame2numpy(frame, (800, 600))
                    print(image)
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
