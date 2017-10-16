from keras.models import load_model
from model import *
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy
from deepgtav.client import Client
import win32gui
import cv2
import numpy as np
from PIL import ImageGrab
from preprocessing import normalize

model_path = 'model_checkpoint420.h5'
# -----------------     drive 환경을 정의하는 변수들    ----------------- #
weather = 'EXTRASUNNY'  # 날씨
vehicle = 'blista'  # 차량
time = [12, 0]  # 시간
drivingMode = [786603, 40.0]  # 운전모드 [mode flag, maximum speed]
location = [-2573.13916015625, 3292.256103515625, 13.241103172302246]  # 시작 위치
frame = [600, 800]  # 화면 크기 <중요> 이거 원래는 [600, 800] 이어야 맞는건가?
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


print("Loading Model...")
model = load_model(model_path)
print("Model Loaded. Compiling...")
sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)  # SGD Optimizer 사용
model.compile(optimizer=sgd, loss="mse")  # loss function 은 논문을 따라 "mean squared error"
model.summary()

if input("Continue?") == "y": # Wait until you load GTA V to continue, else can't connect to DeepGTAV
    print("Conintuing...")

# Loads into a consistent starting setting
print("Loading Scenario...")
client = Client(ip='localhost', port=8000) # Default interface
scenario = Scenario(weather=weather, vehicle=vehicle, time=time, drivingMode=-1,location=location)
client.sendMessage(Start(scenario=scenario))

hwnd_list = _get_windows_bytitle("Grand Theft Auto V", exact=True)  # window 를 받아온다.

count = 0
print("Starting Loop...")
while True:
    try:
        # Collect and preprocess image
        message = client.recvMessage()
        # window title 을 이용해 받아온 window 에서 frame 을 따온다.
        # 기존의 기능에서 이를 제대로 지원하지 않는듯 하기 때문에 추가로 작성함.
        frame_image = screenshot(hwnd=hwnd_list[0])
        # frame_image = normalize(frame_image)
        frame_image = ((frame_image / 255) - .5) * 2  # Simple preprocessing

        # Corrects for model input shape
        frame_image = image.img_to_array(frame_image)
        frame_image = np.reshape(frame_image, (1,) + frame_image.shape)

        prediction = model.predict(frame_image)
        steering = prediction[0][0]
        steering = float(steering)
        print("predicted steering : " + str(steering))
        client.sendMessage(Commands(0.0, 0.0, steering=steering*3))
        # Mutiplication scales decimal prediction for harder turning
        count += 1

    except Exception as e:
        print("Excepted as: " + str(e))
        continue

client.sendMessage(Stop())  # Stops DeepGTAV
client.close()
