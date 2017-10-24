import pickle
import gzip
from deepgtav.messages import frame2numpy


dataset_path = 'F:/augmented_data.pz'


def load_batches_numeric(verbose=1, samples_per_batch=100):
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

                    image = frame2numpy(data_dct['frame'], (800, 600))

                    steering = data_dct['steering']

                    # Train test split
                    # TODO: Dynamic train test split | Test series at end of batch
                    if (count % 10) != 0:  # Train
                        x_train.append(image)
                        y_train.append(steering)
                    else:  # Test
                        x_test.append(image)
                        y_test.append(steering)
                    
                    count += 1
                    if (count % 50) == 0 and verbose == 1:
                        print('     ' + str(count) + ' data points loaded in batch.')
            print('Batch loaded.')
            batch_count += 1
            yield x_train, y_train, x_test, y_test
        except EOFError:  # Breaks at end of file
            break


def load_batches_category(verbose=1, samples_per_batch=1000):
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

                # 아래 Simple preprocessing 은 그냥 down sizing 인듯 하다.
                # image normalization 을 하면 필요 없을듯 하다. <중요>
                image = frame2numpy(data_dct['frame'], (800, 600))
                # image = normalize(image)  # Normalization
                image = ((image / 255) - .5) * 2  # Simple preprocessing

                steering = data_dct['steering']


                # Train test split
                # TODO: Dynamic train test split | Test series at end of batch
                if (count % 10) != 0:  # Train
                # if count != 10:
                    x_train.append(image)
                    # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
                    # y_train.append(int(float(data_dct['steering']) * 500) + 500)  # for categorical
                    y_train.append(int(float(steering) * 500) + 500)  # for numeric
                else:  # Test
                    x_test.append(image)
                    # SantosNet 에서는 steering 을 1~1000 의 정수로 categorize 하였음.
                    # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
                    # y_test.append(int(float(data_dct['steering']) * 500) + 500)
                    y_test.append(int(float(steering) * 500) + 500)  # for numeric

                count += 1
                if (count % 250) == 0 and verbose == 1:
                    print('     ' + str(count) + ' data points loaded in batch.')
            print('Batch loaded.')
            batch_count += 1
            yield x_train, y_train, x_test, y_test
        except EOFError:  # Breaks at end of file
            break