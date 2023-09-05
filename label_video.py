import torch
from torchvision import transforms
import sys
import os
from ultralytics import YOLO
import joblib

import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import argparse


def three_vector_angle(a, b, c):
    
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

def put_text_on_video_hand(drawn_img,alert,fps,frame_count):
    global k
    if ("bending" in alert) :
        k = int(fps)
        print("k : ",k)
        drawn_img =  cv2.putText(drawn_img, f"{frame_count} : Bending",(120,146), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (0, 0, 255), 4)
    else:
        print("k : ",k)
        k-= 1
        if k>1:
            drawn_img = cv2.putText(drawn_img,f"{frame_count} : Bending",(120,146), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (0, 0, 255), 4)
        
        else:
            if ("standing" in alert) :
                drawn_img =   cv2.putText(drawn_img, "Standing",(120,146), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (0, 255, 0), 4)
                return drawn_img
        

    
    return drawn_img

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
    
    final_features = np.array([])
    
    feature_cols = [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER',
        'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_KNEE', 'RIGHT_KNEE',
    ]
    
    box_x1 = (rp[0]["LEFT_HIP"][0] + rp[0]["RIGHT_HIP"][0]) / 2.0
    box_y1 = (rp[0]["LEFT_HIP"][1] + rp[0]["RIGHT_HIP"][1]) / 2.0
    # box_w = max(0.01, np.sqrt((rp[0]["LEFT_HIP"][0] - rp[0]["RIGHT_HIP"][0])**2 + (rp[0]["LEFT_HIP"][1] - rp[0]["RIGHT_HIP"][1])**2))
    # box_h = max(0.01, np.sqrt((rp[0]["LEFT_HIP"][0] - rp[0]["RIGHT_HIP"][0])**2 + (rp[0]["LEFT_HIP"][1] - rp[0]["RIGHT_HIP"][1])**2))

    
    for key in feature_cols:
        x0 = rp[0][key][0]
        y0 = rp[0][key][1]
        
        distance = cal_distance([x0,y0],[box_x1,box_y1])
        final_features = np.append(final_features,[distance])

    shoulder_x, shoulder_y = midpoint_of(rp[0],"LEFT_SHOULDER","RIGHT_SHOULDER")
    hip_x, hip_y = midpoint_of(rp[0], "LEFT_HIP", "RIGHT_HIP")
    knee_x, knee_y = midpoint_of(rp[0], "LEFT_KNEE", "RIGHT_KNEE")
    ankle_x, ankle_y = midpoint_of(rp[0], "LEFT_ANKLE", "RIGHT_ANKLE")
    
    shoulder_coords = [int(shoulder_x),int(shoulder_y)]
    hip_coords = [int(hip_x),int(hip_y)]
    knee_coords = [int(knee_x),int(knee_y)]
    ankle_coords = [int(ankle_x),int(ankle_y)]
    
    angle_one = float(three_vector_angle(hip_coords, knee_coords, ankle_coords)) 
    angle_two = float(three_vector_angle(shoulder_coords, hip_coords, ankle_coords))    

    angle_one = 0.5 if np.isnan(angle_one) else angle_one
    angle_two = 0.5 if np.isnan(angle_two) else angle_two

    final_features = np.append(final_features,[new_features(shoulder_coords, knee_coords, ankle_coords, hip_coords)])
    final_features = np.append(final_features,[angle_one, angle_two])

    return final_features

def process_keypoints(relative_pts):

    relative_pts = relative_pts.xy[0]
    
    updated_keypoints = {"LEFT_SHOULDER" : [int(relative_pts[5][0]),int(relative_pts[5][1])],
                        "RIGHT_SHOULDER": [int(relative_pts[6][0]),int(relative_pts[6][1])],
                        "LEFT_HIP" : [int(relative_pts[11][0]),int(relative_pts[11][1])],
                        "RIGHT_HIP": [int(relative_pts[12][0]),int(relative_pts[12][1])],
                        "LEFT_ANKLE": [int(relative_pts[15][0]),int(relative_pts[15][1])],
                        "RIGHT_ANKLE": [int(relative_pts[16][0]),int(relative_pts[16][1])],
                        "LEFT_KNEE" : [int(relative_pts[13][0]),int(relative_pts[13][1])],
                        "RIGHT_KNEE": [int(relative_pts[14][0]),int(relative_pts[14][1])]
                        }

    return updated_keypoints

def get_kps(relative_kps_all):

    relative_kps_new = []
    # print(relative_kps_all.keypoints)

    for i,relative_kps in enumerate(relative_kps_all.keypoints):
        # print("=======>>",relative_kps)
        try:
            relative_kps_new += process_keypoints(relative_kps),
        except:
            pass
        
    return relative_kps_new

def draw_kps(image,person_mapping):
    # print("person_mapping : ",person_mapping)

    for body_part,ls in person_mapping.items():
        
        x,y = ls

        if body_part in {"LEFT_SHOULDER","LEFT_HIP","LEFT_ANKLE","LEFT_KNEE"}:
            image = cv2.circle(image, (x,y), 12, (0,255,0), -1)
        if body_part in {"RIGHT_SHOULDER","RIGHT_HIP","RIGHT_ANKLE","RIGHT_KNEE"}:
            image = cv2.circle(image, (x,y), 12, (0,0,255), -1)

    return image

def change_dims(h,w):
    
    height = 32*(h//32) if h%32==0 else 32*(h//32+1)
    width = 32*(w//32) if w%32==0 else 32*(w//32+1)
 
    return height, width



# ARGS PARSER----------------------------------------------------------
k=0
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='source') 
parser.add_argument('--action', help='what operation to perform')
args = parser.parse_args()

# Get the width and height of the video--------------------------------
cap = cv2.VideoCapture(args.video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

HEIGHT, WIDTH = change_dims(height,width)

# Counters-------------------------------------------------------------
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
fps =0

# Check FPS of the video-----------------------------------------------
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')    
if int(major_ver)  < 3 :
    fps1 = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print ("Frames per second: {0}".format(fps))
else :
    fps1 = cap.get(cv2.CAP_PROP_FPS)
print ("Frames per second : {0}".format(fps))

# Import MODELS--------------------------------------------------------
MODEL_v8 = YOLO("yolov8n-pose.pt")
MODEL_KP = joblib.load('./models/bending_model_v3.joblib')


for idx, video_path in enumerate([args.video]):

    print("video_path : ",video_path)
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    
    vid_name = os.path.basename(video_path)

    # out = cv2.VideoWriter(f'output_{vid_name}',fourcc,25, (WIDTH,HEIGHT))
    out = cv2.VideoWriter(f'output/output_{vid_name}',fourcc ,25, (WIDTH,HEIGHT)) 

    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    frame_count = 1 # To count total frames.
    total_fps = 0 # To get the final frames per second.
    iou_threshold = 0.65
    confidence_threshold = 0.65

    while(cap.isOpened):
    # Capture each frame of the video.
        ret, frame = cap.read()
        if ret:
            # import pdb;pdb.set_trace();

            print("Frame processing : "+str(frame_count))
            
            frame = cv2.resize(frame,(WIDTH,HEIGHT))

            start1 = time.time()

            start = time.time()
            result = MODEL_v8.predict(frame,imgsz=(WIDTH,HEIGHT),conf=0.25,iou=0.65)
            end=time.time()
            fps_yolo = 1/(end-start)
            
            print("Yolo results : ",len(result))

            output_mapping = get_kps(result[0])
            print("model_1_fps",fps_yolo)

            frame = result[0].plot()

            # cv2.imwrite("./output/"+str(frame_count)+".jpg",frame)

            # relative_points = convert(rp)
            
            if not output_mapping:
                print("No person detected")
                frame_count+=1
                continue
                #print("no person in frame ")

            else:

                class_names = []     

                start=time.time()
                for person_mapping in output_mapping:
                    frame = draw_kps(frame,person_mapping)
                    X_test = preprocess([person_mapping])
                    class_names+= MODEL_KP.predict(X_test.reshape(1, -1))[0],
                end=time.time()

                fps_model = 1/(end-start)

                print("model_2_fps: ",fps_model)
                print("class names --> ",class_names)

                x,y,w,h = [70,105,350,100]

                if "bending" in class_names:
                    print("=======================================================>> Person bent")

                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),-1)
                    frame =  cv2.putText(frame, f"Bending",(x+40,y+70), cv2.FONT_HERSHEY_SIMPLEX,
                                                2, (255,255,255), 4)
                elif "standing" in class_names:
            
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),-1)
                    frame =   cv2.putText(frame, "Standing",(x+40,y+70), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255,255,255), 4)

                # image = put_text_on_video_hand(frame,class_names,fps1,frame_count)
            # Write the FPS on the current frame.

            end1 = time.time()
            
            overall_fps = 1/(end1-start1)
            print("Overall fps: ",overall_fps)
            
            frame = cv2.putText(frame, f"{overall_fps:.2f} FPS", (x,y) , cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            
            
            # Add fps to total fps.
            total_fps += overall_fps
            
            print(" ")
            print(" ")

            out.write(frame)
            
            frame_count += 1

            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    print("Video processing is completed")
    cap.release()  
    out.release()                             
    cv2.destroyAllWindows()
    
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")



