"""! @brief module responsible for launching video/camera, and applying the general features model."""
from ultralytics import YOLO
import cv2
import json


try:
    with open('main_config.json') as f:
        vehicle_det_config = json.load(f)
except :
    raise FileNotFoundError("The file 'main_config.json' was not found.")



def initialize_components(path):
    """Initializes necessary components for object detection.

    @return A tuple containing multiple values: 
        - model 
        - frame_number  
        - the results of the general features (car/truck, lp, and brand) model
    """
    model = YOLO(vehicle_det_config['general_features_model'])
    cap = cv2.VideoCapture(path) 
    #cap = cv2.VideoCapture(0)   
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()  # Create background subtractor object
    return model, cap, bg_subtractor

def object_detection_loop(model, cap, frame_number):
    """Perform object detection on video frames.

    @param the yolo model that identifies the general features (car/truck, lp, and brand) and cap.

    @return the frame's content, frame_number ,and the results of the general features model
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            raise Exception("Error: Frame could not be read.")

        frame_number += 1
        results = model.predict(frame)
        return frame, results, frame_number
    