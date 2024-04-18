"""! @brief A Python program that identifies specific features of a given vehicle."""
##
# @mainpage Vehicle Recognition  Project
#
# @section vehicle_detector Description
# A Python program that detects vehicles entering a parking lot, 
# identifies their types (car/truck), brands, colors, registrations, and nationalities 
##
# @file vehicle_detector.py
#
# @section libraries_vehicle_detector Libraries/Modules
# - random standard library (https://docs.python.org/3/library/random.html)
#   - Access to randint function.
# - PyTorch library (https://pytorch.org/)
# - JSON encoding and decoding (https://docs.python.org/3/library/json.html)
# - UUID objects according to RFC 4122 (https://docs.python.org/3/library/uuid.html)
# - Miscellaneous operating system interfaces (https://docs.python.org/3/library/os.html)
# - OpenCV (https://docs.opencv.org/4.x/index.html)
# - EasyOCR library for Optical Character Recognition (OCR) (https://github.com/JaidedAI/EasyOCR)
# - NumPy for numerical computing (https://numpy.org/)
# - Python Imaging Library (PIL) (https://pillow.readthedocs.io/en/stable/)
# - TorchVision for computer vision tasks with PyTorch (https://pytorch.org/vision/stable/index.html)
# - Ultralytics YOLO (You Only Look Once) object detection (https://github.com/ultralytics/ultralytics)
# - Eclipse Paho MQTT Python client (https://www.eclipse.org/paho/index.php?page=clients/python/index.php)
#
#
# @section notes_vehicle_detector Notes
# - Comments are Doxygen compatible.
#
# @section todo_vehicle_detector TODO
# - a minor work is still needed on the european countries.
#
# 
import cv2
import json
import vehicle_color
import vehicle_license
import vehicle_initialize_detection
import vehicle_features
import vehicle_mqtt

try:
    with open('main_config.json') as f:
        vehicle_det_config = json.load(f)
except :
    raise FileNotFoundError("The file 'main_config.json' was not found.")

file_path = vehicle_det_config['video_path']

def final_features_optimized(video_file):
    """!Identifies the needed features of the vehicle.
    
     @param the video's path or webcam.
    
    @return all the required features in a list in the following form ['car'/'truck', 'LP', 'brand' , 'nationality' , 'color']
    """
    # Initialize components
    model, cap, frame_number, bg_subtractor = vehicle_initialize_detection.initialize_components(file_path)

    while True: 
        # Object detection loop
        frame, results, frame_number = vehicle_initialize_detection.object_detection_loop(model, cap)
        final_features=[]
        # Filter and process detected objects
        final_features = vehicle_features.filter_process_objects(results)
        while final_features is not None:
            # License plate recognition
            nationality = vehicle_license.recognize_license_plate(results, frame_number)
            final_features.append(nationality[1])
            final_features[1] = nationality[0]
            area_threshold = 200
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)
            fg_mask = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)[1]
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # color identification
            color = vehicle_color.identify_vehicle_color(frame, contours, area_threshold, frame_number)
            final_features.append(color)

            # Return final features if all required features are found
            if len(final_features) == 5:
                return final_features

while True:
    features_list = final_features_optimized(file_path)
    vehicle_mqtt.run(features_list)