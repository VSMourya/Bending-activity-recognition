# Bending_activity_recognition

The "Human Activity Recognition" project utilizes the YOLOv8 model for keypoint detection and a Random Forest classifier to recognize the "person_bending" action. The project consists of two pipelines:

1."Make_alerts.py" generates real-time alert videos from a live camera feed, saving them in the "alert_vids" folder.

2."Label_video.py" labels a given video for the specific bending action. The input video is taken from the "input" folder, and the labelled video is saved in the "output" folder.

To provide context in the alert videos, a personal cache is implemented, which stores multiple frames before generating the alert. This ensures that the final alert video includes a few seconds of frames before the action occurs.

To facilitate easy deployment and reproducibility, the project includes a Dockerfile to create a Docker image. Running the project inside a Docker container ensures consistent and isolated execution of the code.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/VSMourya/Bending_activity_recognition.git
   ```
2. Create a Docker image using the below command
   ```
   Docker build -t <image_name>:<tag_name> .
   ```
3. Build a docker container from docker Image
   ```
   docker create -v <current_directory_path>/:/<folder_name:>  --name <container_name>  <Image_ID>
   ```
4. Run the Docker Container
   ```
   docker exec -i -t 
   ```
   ```
   cd <folder_name>
   ``` 
5. Install all the requirements needed
   ```
   pip install -r requirements.txt
   ```

6. Run either of the commands with input folder container the video input.mp4
   ```
   python3 make_alerts.py -video input.mp4
   ```
   ```
   python3 label_video.py -video input.mp4
   ```