# Project_1080 : create_dataset.py
# Reference : https://github.com/cpgeier/SantosNet

from deepgtav.messages import Start, Stop, Dataset, Scenario, Config
from deepgtav.client import Client
import argparse
import datetime
import win32gui
import win32ui
import win32con
from time import sleep
from PIL import ImageGrab
import cv2
import numpy as np


# window title 로 해당 window 를 가져오는 함수
def _get_windows_bytitle(title_text, exact=False):
    def _window_callback(hwnd, all_windows):
        all_windows.append((hwnd, win32gui.GetWindowText(hwnd)))
    windows = []
    win32gui.EnumWindows(_window_callback, windows)
    if exact:
        return [hwnd for hwnd, title in windows if title_text == title]
    else:
        return [hwnd for hwnd, title in windows if title_text in title]


# _get_windows_bytitle 을 사용해 가져온 window 를 넘겨받아 frame 을 따오는 함수
def screenshot(hwnd=None):
    if not hwnd:
        hwnd = win32gui.GetDesktopWindow()

    rect = win32gui.GetWindowRect(hwnd)
    img = ImageGrab.grab(bbox=rect)
    img = np.array(img)
    bgr = img[..., ::-1]

    return bgr


# -----------------     Dataset 을 수집할 환경을 정의하는 변수들    ----------------- #
weather = 'EXTRASUNNY'  # 날씨
vehicle = 'blista'  # 차량
time = [12, 0]  # 시간
drivingMode = [786603, 40.0]  # 운전모드 [mode flag, maximum speed]
location = [-2573.13916015625, 3292.256103515625, 13.241103172302246]  # 시작 위치
frame = [800, 600]  # 화면 크기 <중요> 이거 원래는 [600, 800] 이어야 맞는건가?
dataset_path = 'dataset_test.pz'
#  ---------------------------------------------------------------------------------  #

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
    Client.sendMessage(Config(scenario=scenario, dataset=dataset).to_json())


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
    dataset = Dataset(rate=30, frame=frame, throttle=True, brake=True, steering=True, location=True, drivingMode=True,
                      peds=True, reward=[15.0, 0.0], direction=None, speed=True, yawRate=True, time=True)

    # Automatic driving scenario - Dataset 을 구성할 Driving scenario 를 정의한다.
    scenario = Scenario(weather=weather, vehicle=vehicle, time=time, drivingMode=drivingMode, location=location)

    # GTAV 에 dataset path 와 scenario 를 전달하여 dataset 을 모으기 위한 환경을 실행한다.
    client.sendMessage(Start(dataset=dataset, scenario=scenario))  # Start request
    hwnd_list = _get_windows_bytitle("Grand Theft Auto V", exact=True)  # window 를 받아온다.

    # count 는 현재까지 받아온 frame 의 수를 의미한다.
    count = 0
    old_location = [0, 0, 0]

    started = False  # frame 을 받아오는 것이 시작되었는지 여부 flag
    start_time = datetime.datetime.now()  # 시작 시간, 우선 초기화

    while True:  # Main Loop
        current_time = datetime.datetime.now()
        if started is True and current_time >= start_time + datetime.timedelta(hours=3):
            print("Finished recording at " + str(current_time))
            break

        try:
            # Message received as a Python dictionary - 실질적으로 data 를 읽어온다.
            # rate 관리도 여기서 일어나는 듯 하다.
            message = client.recvMessage_notSave()
            # window title 을 이용해 받아온 window 에서 frame 을 따온다.
            # 기존의 기능에서 이를 제대로 지원하지 않는듯 하기 때문에 추가로 작성함.
            frame_img = screenshot(hwnd=hwnd_list[0])

            # datafile 에 저장한다!!!
            client.save_to_datafile(frame_img, message)

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
                    print("reset location at " + str(current_time))
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



