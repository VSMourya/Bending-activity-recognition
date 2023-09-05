import joblib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import argparse
from utilities import check_bending_action_yolov8
import config_file

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='source') 
args = parser.parse_args()

frame_count = 0
model_kp = joblib.load(config_file.MODEL_PATH)


if args:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)


if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

while(cap.isOpened):
# Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        frame_count+=1
        
        start = time.time()
        alert, (alert_video_path,success) = check_bending_action_yolov8(frame,frame_count,model_kp,show_img=False)
        end = time.time()

        print("FPS_each_frame : ",round(1/(end-start),3))
        print(" ")
        print(" ")
    else:
        break

cap.release()  