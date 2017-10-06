# Project_1080 : create_dataset.py
# Reference : https://github.com/cpgeier/SantosNet

from deepgtav.messages import Start, Stop, Dataset, Scenario, Config
from deepgtav.client import Client
import argparse
import datetime
import cv2

# Dataset 을 수집할 환경을 정의하는 변수들
weather = 'EXTRASUNNY'  # 날씨
vehicle = 'blista'  # 차량
time = [12, 0]  # 시간
drivingMode = [786603, 40.0]  # 운전모드 [mode flag, maximum speed]
location = [-2573.13916015625, 3292.256103515625, 13.241103172302246]  # 시작 위치
frame = [800, 600]  # 화면 크기
dataset_path = 'dataset.pz'

""" 
자세한 flag 정보는 http://gtaforums.com/topic/822314-guide-driving-styles/ 참고
----------------------  DRIVING MODES  ----------------------
FLAGS - CONVERTED INTEGER - NAME/DESC OF THE DRIVING STYLE
01000000000011000000000000100101 - 1074528293 -  "Rushed"
00000000000011000000000000100100 - 786468 - "Avoid Traffic"
00000000000000000000000000000110 - 6 - "Avoid Traffic Extremely"
00000000000011000000000010101011 - 786603 - "Normal"
00000000001011000000000000100101 - 2883621 - "Ignore Lights"
"""


def reset():
    # Resets position of the car to the starting location
    dataset = Dataset(rate=30, frame=frame, throttle=True, brake=True, steering=True, location=True, drivingMode=True)
    scenario = Scenario(weather=weather, vehicle=vehicle, time=time, drivingMode=drivingMode, location=location)
    Client.sendMessage(Config(scenario=scenario, dataset=dataset))


# Stores a picked dataset file with data coming from DeepGTAV
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-d', '--dataset_path', default=dataset_path, help='Place to store the dataset')
    args = parser.parse_args()

    # Create a new connection to DeepGTAV using the specified IP and Port
    client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9)
    # Dataset options
    dataset = Dataset(rate=30, frame=frame, throttle=True, brake=True, steering=True, location=True, drivingMode=True, peds=True, reward=[15.0, 0.0], direction=None, speed=True, yawRate=True, time=True)
    # Automatic driving scenario
    scenario = Scenario(weather=weather, vehicle=vehicle, time=time, drivingMode=drivingMode, location=location)
    client.sendMessage(Start(dataset=dataset, scenario=scenario))  # Start request

    # count 는 현재까지 받아온 frame 의 수를 의미한다.
    count = 0
    old_location = [0, 0, 0]

    started = False
    start_time = datetime.datetime.now()

    while True:  # Main Loop
        current_time = datetime.datetime.now()
        if current_time >= start_time + datetime.timedelta(hours=3):
            print("Finished recording at " + str(current_time))
            break

        try:
            # Message received as a Python dictionary
            message = client.recvMessage()
            del message['frame']
            print(message)
            if started is False:
                start_time = datetime.datetime.now()
                print("Start recording at " + str(start_time))
                started = True

            if (count % 100) == 0:
                print(str(count) + " frames are recorded.")

            # Checks if car is sucked, resets position if it is.
            new_location = message['location']
            # Float position converted to ints so it doesn't have to be in the exact same position to be reset
            if (count % 1000) == 0:
                if (int(new_location[0]) == int(old_location[0]) and int(new_location[1]) == int(old_location[1]) and
                        int(new_location[2]) == int(old_location[2])):
                    reset()

                old_location = message['location']
                # print('At location: ' + str(old_location))
            count += 1

        except KeyboardInterrupt:
            i = input('Paused. Press p to continue and q to exit...')
            if i == 'p':
                continue
            elif i == 'q':
                break

    print("Total frames recored : " + str(count))
    # DeepGTAV stop message
    client.sendMessage(Stop())
    client.close()



