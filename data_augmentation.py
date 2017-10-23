import pickle
import gzip
import cv2
import numpy as np

from deepgtav.messages import frame2numpy
import matplotlib.pyplot as plt

import time
import datetime

dataset_path = 'dataset.pz'
dataset = gzip.open(dataset_path)

save_to = 'F:/augmented_data.pz'  # save augmented data to this path
pickleFile = gzip.open(save_to, mode='ab', compresslevel=9)
values = []  # steering 값들을 저장하는 리스트 - 그래프 그리는 용도

count = 0
augmented_count = 0
abandon_count = 0
near_zero = 0

# record start time
ts = time.time()
start_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

while True:
    count += 1
    print("Processing %d images ... " % count)

    try:
        data_dct = pickle.load(dataset)
        steering = data_dct['steering']
        values.append(steering)

        # data augmentation
        # steering value 의 절대값이 0.0625 이상이면 augmentation을 한다.
        # [원본 이미지 - steering] 과 +- 0.00001, 좌우 반전을 하여 6배로 부풀린다.
        if abs(steering) > 0.0625:
            augmented_count += 5

            # augment data
            image = data_dct['frame']
            image = frame2numpy(data_dct['frame'], (800, 600))
            pickle.dump(data_dct, pickleFile)  # dump original image and steering value

            original_steering = data_dct['steering']

            data_dct['steering'] = original_steering - 0.00001
            pickle.dump(data_dct, pickleFile)  # dump original image with distorted steering value 1
            values.append(data_dct['steering'])

            data_dct['steering'] = original_steering + 0.00001
            pickle.dump(data_dct, pickleFile)  # dump original image with distorted steering value 2
            values.append(data_dct['steering'])

            # left-right flip
            original_steering = original_steering * (-1)  # sign change
            image = cv2.flip(image, 1)  # flip original image vertically

            data_dct['frame'] = image
            data_dct['steering'] = original_steering
            pickle.dump(data_dct, pickleFile)  # dump flipped image with sign change
            values.append(data_dct['steering'])

            data_dct['steering'] = original_steering - 0.00001
            pickle.dump(data_dct, pickleFile)  # dump flipped image with distorted steering value 1
            values.append(data_dct['steering'])

            data_dct['steering'] = original_steering + 0.00001
            pickle.dump(data_dct, pickleFile)  # dump flipped image with distorted steering value 2
            values.append(data_dct['steering'])

        else:  # -0.0625 ~ 0.0625 사이의 값은 충분히 많다고 판단하여 반으로 줄인다.
            near_zero += 1

            if near_zero == 2:
                pickle.dump(data_dct, pickleFile)
                near_zero = 0
            else:
                abandon_count += 1

    except [IOError, 28]:
        print("No space left on the device!!!")
        break

    except EOFError:
        break

ts = time.time()
end_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

print("Data Augmentation Finished!!")
print("Started at " + str(start_time))
print("Ended at " + str(end_time))
print("# of data before : %d" % count)
print("# of data after : %d" % (count - abandon_count + augmented_count))

plt.hist(values, bins=200)
plt.title("steering values distribution")
plt.xlabel("Steering Value")
plt.ylabel("Count")
plt.show()


