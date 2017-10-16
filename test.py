import pickle
import gzip
import cv2

dataset_path = 'dataset.pz'
dataset = gzip.open(dataset_path)
count = 0
while True:
    try:
        data_dct = pickle.load(dataset)
        count += 1

    except EOFError:
        break

print(count)


