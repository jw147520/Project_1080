import win32gui
import win32ui
import win32con
from time import sleep
from PIL import ImageGrab
import cv2
import numpy as np


def screenshot(hwnd=None):
    if not hwnd:
        hwnd = win32gui.GetDesktopWindow()

    rect = win32gui.GetWindowRect(hwnd)
    img = ImageGrab.grab(bbox=rect)
    img = np.array(img)
    bgr = img[..., ::-1]
    print(bgr)
    cv2.imshow('img', bgr)
    cv2.waitKey(0)


def _get_windows_bytitle(title_text, exact=False):
    def _window_callback(hwnd, all_windows):
        all_windows.append((hwnd, win32gui.GetWindowText(hwnd)))
    windows = []
    win32gui.EnumWindows(_window_callback, windows)
    if exact:
        return [hwnd for hwnd, title in windows if title_text == title]
    else:
        return [hwnd for hwnd, title in windows if title_text in title]


hwnd_list = _get_windows_bytitle("Grand Theft Auto V", exact=True)

print(len(hwnd_list))


screenshot(hwnd=hwnd_list[0])
