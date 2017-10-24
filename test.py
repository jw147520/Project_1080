import pickle
import gzip
import cv2
from deepgtav.messages import frame2numpy
import numpy as np

dataset_path = 'dataset.pz'
save_to = 'F:/dataset.pz'
dataset = gzip.open(dataset_path)

# pickleFile = gzip.open(save_to, mode='ab', compresslevel=9)

count = 0
while True:

    if count > 300:
        break
    try:
        data_dct = pickle.load(dataset)
        print(data_dct['steering'])
        image = data_dct['frame']
        image = frame2numpy(data_dct['frame'], (800, 600))
        image = ((image / 255) - .5) * 2
        print(type(image))
        # pickle.dump(data_dct, pickleFile)


        # print(image)
        # print(image.shape)

        print(count)
        count += 1

    except EOFError:
        break

print(count)


