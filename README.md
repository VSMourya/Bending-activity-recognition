# Bending_activity_recognition

The "Human Activity Recognition" project utilizes the YOLOv8 model for keypoint detection and a Random Forest classifier to recognize the "person_bending" action. The project consists of two pipelines:

1."Make_alerts.py" generates real-time alert videos from a live camera feed, saving them in the "alert_vids" folder.

2."Label_video.py" labels a given video for the specific bending action. The input video is taken from the "input" folder, and the labelled video is saved in the "output" folder.

To provide context in the alert videos, a personal cache is implemented, which stores multiple frames before generating the alert. This ensures that the final alert video includes a few seconds of frames before the action occurs.

To facilitate easy deployment and reproducibility, the project includes a Dockerfile to create a Docker image. Running the project inside a Docker container ensures consistent and isolated execution of the code. 