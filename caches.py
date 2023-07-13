import cv2
from collections import deque


class video_cache():

    def update_values(self):
        
        #reset all parameters and release the videowriter
        self.video_creation_started = False
        self.prev_images_loaded = False
        self.frame_counter = 50
        self.counter = 0
        self.already_alert = False
        self.alert=[]

        return 

    def prepare_video(self,frame_no,img):

        if self.prev_images_loaded:
            self.counter +=1
            # print("counter : ",self.counter , " frame_counter : ", self.frame_counter)
            if self.counter != self.frame_counter:
                self.result.write(img)
                self.pre_alert_cache.append(img)
            else:
                print("---------------------------------------------------->>> Ended video")
                self.result.release()
                self.update_values()
        else:
            self.video_path = f'alert_vids/bending_{frame_no}.webm'
            self.result = cv2.VideoWriter(self.video_path, 
                         cv2.VideoWriter_fourcc('V','P','8','0'),
                         self.vid_fps, self.size)
            print("---------------------------------------------------->>> Started making video")

            self.prev_images_loaded = True
            for each_img in self.pre_alert_cache:
                self.result.write(each_img)

            self.prev_images_loaded = True
            return self.video_path , True

        return None, False

    def update_video_cache(self,frame_no,img,alert=False):
        
        height, width, _ = img.shape
        self.size = (width,height)
        
        if alert:
        
            if self.already_alert:
                return self.prepare_video(frame_no,img)
            self.alert=alert
            self.already_alert = True
            if self.video_creation_started:
                self.prev_images_loaded = True
            else:
                self.video_creation_started = True
            
            return self.prepare_video(frame_no,img)

        else:
            
            if self.video_creation_started:
                return self.prepare_video(frame_no,img)
            
            if self.frame_counter == 0:
                self.video_creation_started = False

            self.pre_alert_cache.append(img)  

        return None, False
    
    def __init__(self):
        
        self.vid_fps = 20
        self.pre_alert_cache = deque(maxlen=(self.vid_fps)*3)
        self.frame_counter = ((self.vid_fps)*3)-1
        self.already_alert = False
        self.counter = 0
        self.video_creation_started = False
        self.prev_images_loaded = False
        self.alert=[]

class bending_alert_cache():
    
    '''
    Instance of this class mantains 
    '''
    
    def check_bending(self,status):

        if self.len1 == 3:
            self.cache_bending.popleft()
            self.len1 -= 1

        self.cache_bending.append(status)            
        self.len1 += 1
        
        return all(self.cache_bending)

    def update_cache(self,bending_status):
        
        ret = []
        
        bent_current_seq = self.check_bending(bending_status)

        if bent_current_seq and (not self.prev_bent):
            ret.append('person_bent')

        self.prev_bent = bent_current_seq

        return ret

    def __init__(self):
        self.cache_bending = deque([])
        self.len1 = 0
        self.prev_bent = False
