import cv2
import gzip
import pickle
from deepgtav.messages import frame2numpy

dataset = gzip.open('dataset_test.pz')
print(dataset)

data_dct = pickle.load(dataset)

count = 0
while True:
    try:
        frame = data_dct['frame']
        image = frame2numpy(frame, (800, 600))
        cv2.imshow('img', image)
        cv2.waitKey(-1)
        count += 1

    except EOFError:
        break

print(count)

