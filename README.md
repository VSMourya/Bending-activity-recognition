# Bending Activity Recognition

## Introduction
"Bending Activity Recognition" is an innovative project leveraging YOLOv8 model for keypoint detection and a Random Forest classifier. Detecting bending actions is crucial in critical environments, including pharmaceuticals, to ensure safety and hygiene, prevent contamination risks, and maintain adherence to strict operational protocols.

## System Requirements
- Python 3.8 or above
- Required Python libraries (see `requirements.txt`)
- Docker (optional for containerized deployment)

## Installation Steps
1. Clone the repository to your local machine:
   ```sh
   git clone https://github.com/VSMourya/Bending_activity_recognition.git
   ```
2. Install the required Python libraries:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Project
1. Navigate to the cloned project directory.
2. Run the desired script for alert generation or video labeling:
   - For real-time alert generation:
     ```sh
     python3 make_alerts.py -video input.mp4
     ```
   - For labeling videos:
     ```sh
     python3 label_video.py -video input.mp4
     ```

## Files Description
- `make_alerts.py`: Generates real-time alert videos for bending actions.
- `label_video.py`: Labels bending actions in videos.
- `caches.py`: Manages a cache system for storing frames prior to an alert.
- `utilities.py`: Provides utility functions for the project.
- `config_file.py`: Contains configuration settings for the project.
- `Dockerfile`: Instructions for building a Docker image.

## Docker Setup (Optional)
For Docker users, follow these steps to build and run the project in a container:
1. Build the Docker image:
   ```sh
   docker build -t bending-activity-recognition .
   ```
2. Run the Docker container:
   ```sh
   docker run -it --name bending-activity-recognition bending-activity-recognition
   ```

## Additional Information
- The project uses the YOLOv8 model for precise keypoint detection, and a Random Forest classifier for the recognition of bending actions.
- Ensure the video input follows the specified format for optimal results.