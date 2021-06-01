import time
import cv2

def Connect(video_src):
    cap = cv2.VideoCapture(video_src)
    return cap

def Capture(sleep, cap):
    if sleep != 0:
        time.sleep(sleep)
    flag, image = cap.read()
    return flag, image
