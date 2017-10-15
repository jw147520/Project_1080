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
model = load_model('model_checkpoint_3_420.h5')
print("Model Loaded. Compiling...")
sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)  # SGD Optimizer 사용
model.compile(optimizer=sgd, loss="mse")  # loss function 은 논문을 따라 "mean squared error"
model.summary()

if input("Continue?") == "y": # Wait until you load GTA V to continue, else can't connect to DeepGTAV
    print("Conintuing...")

# Loads into a consistent starting setting 
print("Loading Scenario...")
client = Client(ip='localhost', port=8000) # Default interface
scenario = Scenario(weather='EXTRASUNNY',vehicle='blista',time=[12,0],drivingMode=-1,location=[-2573.13916015625, 3292.256103515625, 13.241103172302246])
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
        frame_image = ((frame_image / 255) - .5) * 2  # Simple preprocessing

        # Corrects for model input shape
        frame_image = image.img_to_array(frame_image)
        frame_image = np.reshape(frame_image, (1,) + frame_image.shape)

        # Converts classification to float for steering input
        prediction = model.predict(frame_image)
        # decimal_prediction = (category_prediction - 500) / 500
        # print('Category: ' + str(category_prediction) + '     Decimal: ' + str(decimal_prediction))
        print(prediction[0][0])
        steering = prediction[0][0]
        steering = float(steering)
        client.sendMessage(Commands(0.0, 0.0, steering=steering))
        # Mutiplication scales decimal prediction for harder turning
        count += 1
    except Exception as e:
        print("Excepted as: " + str(e))
        continue

client.sendMessage(Stop())  # Stops DeepGTAV
client.close()
