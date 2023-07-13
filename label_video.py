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

k=0
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='source') 
parser.add_argument('--action', help='what operation to perform')

args = parser.parse_args()

cap = cv2.VideoCapture(args.video)

# Get the width and height of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
fps=0


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3:
    fps1 = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print ("Frames per second: {0}".format(fps))
else :
    fps1 = cap.get(cv2.CAP_PROP_FPS)
print ("Frames per second : {0}".format(fps))

model_v8 = YOLO("yolov8n-pose.pt")

# if torch.cuda.is_available():
#     model.half().to(device)

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
    
    feature_cols = [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER',
        # 'LEFT_ELBOW', 'RIGHT_ELBOW',
        # 'LEFT_WRIST', 'RIGHT_WRIST',
        'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_KNEE', 'RIGHT_KNEE',
    ]
    
    # import pdb; pdb.set_trace()

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
    
    # shoulder_x = (rp["LEFT_SHOULDER"][0] + rp["RIGHT_SHOULDER"][0]) / 2.0
    # shoulder_y = (rp["LEFT_SHOULDER"][1] + rp["RIGHT_SHOULDER"][1]) / 2.0 
   
    # hip_x = (rp["LEFT_HIP"][0] + rp["RIGHT_HIP"][0]) / 2.0
    # hip_y = (rp["LEFT_HIP"][1] + rp["RIGHT_HIP"][1]) / 2.0
    
    # knee_x = (rp["LEFT_KNEE"][0] + rp["RIGHT_KNEE"][0]) / 2.0
    # knee_y = (rp["LEFT_KNEE"][1] + rp["RIGHT_KNEE"][1]) / 2.0

    # ankle_x = (rp["LEFT_ANKLE"][0] + rp["RIGHT_ANKLE"][0]) / 2.0
    # ankle_y = (rp["LEFT_ANKLE"][1] + rp["RIGHT_ANKLE"][1]) / 2.0
    
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

    for i,relative_kps in enumerate(relative_kps_all.keypoints):
        relative_kps_new += process_keypoints(relative_kps),

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

model_kp = joblib.load('./models/bending_model_v3.joblib')

for idx, video_path in enumerate([args.video]):

    print(video_path)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    vid_name = os.path.basename(video_path)

    out = cv2.VideoWriter(f'./output/processed_v8_'+str(vid_name),fourcc,25, (width,height))
    
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

            print("Frame processing : "+str(frame_count))

            start1 = time.time()

            start=time.time()
            result=model_v8.predict(frame,imgsz=(width,height),conf=0.25,iou=0.65)
            end=time.time()
            
            print("result: ",len(result))
            fps_yolo = 1/(end-start)

            output_mapping = get_kps(result[0])
            print("model_1_fps",fps_yolo)


            # Add fps to total fps.
            total_fps += fps

            # img = plot_keypoints(output_mapping,output_score_mapping,frame)
            # img = frame

            frm = result[0].plot()

            # cv2.imwrite("./output/"+str(time.time())+".jpg",frm)
            # cv2.imwrite("./output/"+str(frame_count)+".jpg",frm)

            # relative_points = convert(rp)
            
            # if not output_mapping:
            #     print("No person detected")
            #     continue
            #     #print("no person in frame ")

            # else:

            #     class_names = []     

            #     start=time.time()
            #     for person_mapping in output_mapping:
            #         frm = draw_kps(frm,person_mapping)
            #         X_test = preprocess([person_mapping])
            #         class_names+= model_kp.predict(X_test.reshape(1, -1))[0],
            #     end=time.time()

            #     fps_model = 1/(end-start)

            #     print("model_2_fps: ",fps_model)

            #     # if "bending" in class_names:
            #     #     image_save = "output/"+str(frame_count)+".jpg"
            #     #     cv2.imwrite(image_save,frame)

            #     print("class names --> ",class_names)

            #     x,y,w,h = [110,135,350,100]

            #     if "bending" in class_names:
            #         print("=======================================================>> Person bent")

            #         frm = cv2.rectangle(frm,(x,y),(x+w,y+h),(0,0,255),-1)
            #         frm =  cv2.putText(frm, f"Bending",(x+40,y+70), cv2.FONT_HERSHEY_SIMPLEX,
            #                                     2, (255,255,255), 4)
            #     elif "standing" in class_names:

            #         frm = cv2.rectangle(frm,(x,y),(x+w,y+h),(0,255,0),-1)
            #         frm =   cv2.putText(frm, "Standing",(x+40,y+70), cv2.FONT_HERSHEY_SIMPLEX,
            #                         2, (255,255,255), 4)
                    
                # image = put_text_on_video_hand(frm,class_names,fps1,frame_count)
            # Write the FPS on the current frame.
            # cv2.putText(img, f"{fps:.3f} FPS", (15, 30) , cv2.FONT_HERSHEY_SIMPLEX,
                        # 1, (0, 255, 0), 2)

            end1 = time.time()

            print("Overall fps: ",1/(end1-start1))
            
            print(" ")
            print(" ")

            out.write(frm)
            
            frame_count += 1
            #print("running..",frame_count)


            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:

            break
    # Release VideoCapture().
    print("Video processing is completed")
    cap.release()  
    out.release()                             
    # Close all frames and video windows.
    cv2.destroyAllWindows()

    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")



