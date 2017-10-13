import pickle
import gzip
import cv2

dataset_path = 'dataset.pz'
dataset = gzip.open(dataset_path)
count = 0
while count < 100:

    data_dct = pickle.load(dataset)

    print(data_dct['steering'])
    count += 1
