import torch
from torchvision import transforms
from ultralytics import YOLO

import sys
import joblib

import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from caches import video_cache,bending_alert_cache
import config_file 

vid_cache = video_cache()
alert_cache = bending_alert_cache()

model_v8 = YOLO("models/yolov8n-pose.pt")

def get_angle_abc(a, b, c):
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # np.degrees(angle)
    angle = np.abs(angle)

    return angle

def cal_distance(a,b):

    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b)

def new_features(shoulder_coords, knee_coords, ankle_coords, hip_coords):

    arr = []
    
    arr.append(cal_distance(shoulder_coords, knee_coords))
    arr.append(cal_distance(hip_coords, ankle_coords)) 
    
    return arr

def midpoint_of(rp, bodypart_left, bodypart_right):
    
    x = (rp[bodypart_left][0] + rp[bodypart_right][0]) / 2.0
    y = (rp[bodypart_left][1] + rp[bodypart_right][1]) / 2.0  
    
    return [x,y]

def preprocess(rp):
    
    feature_cols = [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER',
        'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_KNEE', 'RIGHT_KNEE',
    ]
    
    box_x1 = (rp[0]["LEFT_HIP"][0] + rp[0]["RIGHT_HIP"][0]) / 2.0
    box_y1 = (rp[0]["LEFT_HIP"][1] + rp[0]["RIGHT_HIP"][1]) / 2.0
    box_w = max(0.01, np.sqrt((rp[0]["LEFT_HIP"][0] - rp[0]["RIGHT_HIP"][0])**2 + (rp[0]["LEFT_HIP"][1] - rp[0]["RIGHT_HIP"][1])**2))
    box_h = max(0.01, np.sqrt((rp[0]["LEFT_HIP"][0] - rp[0]["RIGHT_HIP"][0])**2 + (rp[0]["LEFT_HIP"][1] - rp[0]["RIGHT_HIP"][1])**2))

    x = np.array([])
    for key in feature_cols:
        x0 = rp[0][key][0]
        y0 = rp[0][key][1]
        
        distance = cal_distance([x0,y0],[box_x1,box_y1])
        x = np.append(x,[distance])

    shoulder_x, shoulder_y = midpoint_of(rp[0],"LEFT_SHOULDER","RIGHT_SHOULDER")
    hip_x, hip_y = midpoint_of(rp[0], "LEFT_HIP", "RIGHT_HIP")
    knee_x, knee_y = midpoint_of(rp[0], "LEFT_KNEE", "RIGHT_KNEE")
    ankle_x, ankle_y = midpoint_of(rp[0], "LEFT_ANKLE", "RIGHT_ANKLE")

    shoulder_coords = [int(shoulder_x),int(shoulder_y)]
    hip_coords = [int(hip_x),int(hip_y)]
    knee_coords = [int(knee_x),int(knee_y)]
    ankle_coords = [int(ankle_x),int(ankle_y)]
    
    l1 = float(get_angle_abc(hip_coords, knee_coords, ankle_coords)) 
    l2 = float(get_angle_abc(shoulder_coords, hip_coords, ankle_coords))    

    l1 = 0.5 if np.isnan(l1) else l1
    l2 = 0.5 if np.isnan(l2) else l2

    x = np.append(x,[new_features(shoulder_coords, knee_coords, ankle_coords, hip_coords)])
    x = np.append(x,[l1, l2])

    return x

def process_keypoints(relative_pts):
    
    updated_keypoints = {"LEFT_SHOULDER" : [int(relative_pts[6][0]),int(relative_pts[6][1])],
                        "RIGHT_SHOULDER": [int(relative_pts[5][0]),int(relative_pts[5][1])],
                        "LEFT_HIP" : [int(relative_pts[12][0]),int(relative_pts[12][1])],
                        "RIGHT_HIP": [int(relative_pts[11][0]),int(relative_pts[11][1])],
                        "LEFT_ANKLE": [int(relative_pts[16][0]),int(relative_pts[16][1])],
                        "RIGHT_ANKLE": [int(relative_pts[15][0]),int(relative_pts[15][1])],
                        "LEFT_KNEE" : [int(relative_pts[14][0]),int(relative_pts[14][1])],
                        "RIGHT_KNEE": [int(relative_pts[13][0]),int(relative_pts[13][1])]
                        }

    return updated_keypoints

def get_kps(relative_kps_all):

    relative_kps_new = []

    for i,relative_kps in enumerate(relative_kps_all):
        relative_kp = np.array(relative_kps.keypoints.data[0])
        try:
            relative_kps_new += process_keypoints(relative_kp),
        except:
            continue

    return relative_kps_new

def change_dims(h,w):
    
    height = 32*(h//32) if h%32==0 else 32*(h//32+1)
    width = 32*(w//32) if w%32==0 else 32*(w//32+1)
 
    return height, width

def check_bending_action_yolov8(frame,frame_count,model_kp,show_img=False):

    print("Running frame number ", frame_count)

    single_frame_alert = False
    multi_frame_alert = False

    # frame = cv2.resize(frame,(864,480))
    
    height,width,_ = frame.shape
    
    height,width = change_dims(height,width)

    start = time.time()
    result = model_v8.predict(frame,imgsz=(width,height),conf=config_file.CONF_THRES,iou=config_file.IOU_THRES)
    end = time.time()
    
    fps_yolo = 1/(end-start)
    print("FPS_yolo : ",round(fps_yolo,3))
    
    output_mapping = get_kps(result)


    if not output_mapping:
        print("no person detected")
        return [],("",False)
    
    else:

        class_names = []
        
        start=time.time()
        for person_mapping in output_mapping:
            X_test = preprocess([person_mapping])
            class_names+= model_kp.predict(X_test.reshape(1, -1))[0],
        end =time.time()
        
        print("FPS_model : ",round(1/(end-start),3))

        frm = result[0].plot()

        print("class names --> ",class_names)

        if "bending" in class_names:
            single_frame_alert = True

        if single_frame_alert:
            multi_frame_alert = alert_cache.update_cache(True)
        else:
            multi_frame_alert = alert_cache.update_cache(False)

        print("SFA : ",single_frame_alert)

        if multi_frame_alert:
            print("Multiframe_alert ========================================>> ",multi_frame_alert)
            return ["Person_bent"],vid_cache.update_video_cache(frame_count,frm,alert=True)
        else:
            print("MFA : ",multi_frame_alert)
            return [],vid_cache.update_video_cache(frame_count,frm,alert=False)
