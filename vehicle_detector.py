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
import torch
import json
import uuid
import os
import cv2
import easyocr
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from paho.mqtt import client as mqtt_client

with open('vehicle_det_config.json') as f:
    vehicle_det_config = json.load(f)

# Functions
def predict_car_color(image_to_cap):
    """! predicts the vehicle color.

    @param image_to_cap  path to the vehicle image.

    @return color of the vehicle.
    """
    final_model = torch.load(vehicle_det_config['color_model'], map_location="cpu")
    # list of the colors that the color model predicts
    colors = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']
    # Define transformations for the input image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and preprocess the input image
    image = Image.open(image_to_cap).convert('RGB')
    image = transform(image).unsqueeze(0)

    #! Perform prediction
    with torch.no_grad():
        preds = final_model(image)
        probabilities = torch.nn.functional.softmax(preds, dim=1)
        top_probability, top_class = probabilities.topk(1, dim=1)

    #! Get the predicted class name 
    car_color = colors[top_class.item()]
    if len(car_color) != 0:
        return car_color 
    else:
        return "silver"

def read_license_plate(im):
    """! reads the license plate content (language is set to english).

    @param image_to_cap  path to the license plate image.

    @return license plate text.
    """
    license_plate_crop=cv2.imread(im)
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
    
    reader = easyocr.Reader(['en'], gpu=False)
    detections = reader.readtext(license_plate_crop)
    # Sort the OCR results by bounding box area and x-coordinate
    sorted_results = sorted(detections, key=lambda x: x[0][0][0])
    lp_text=''
    for detection in sorted_results:
        bbox, text, score = detection
        lp_text+=text.upper()
    return lp_text

def predict_generic_nationality(image_to_cap):
    """! identifies the country out of the license plate.

    @param image_to_cap path to the license plate image.

    @return the country name OR "america" as a default value.
    """
    final_model = YOLO(vehicle_det_config['generic_nationality_model'])
    nationality = ['europe','america','qatar','tunisia','egypt','UAE','libya']

    image = cv2.imread(image_to_cap)
    results = final_model.predict(image)
    class_ids=[]
    confidences=[]
    for result in results:
                boxes = result.boxes.cpu().numpy()
                confidences.append(boxes.conf)
                class_ids.append(boxes.cls)
    if len(boxes) != 0:
        return nationality[int(boxes.cls[0])]
    else:
        return "america"
   
class_names = ['car','truck','LP','Toyota','Volkswagen','Ford','Honda','Chevrolet','Nissan','BMW','Mercedes','Audi','Tesla','Hyundai','Kia','Mazda','Fiat','Jeep','Porsche','Volvo','Land Rover','Peugeot','Renault','Citroen','Isuzu','MAN','Iveco','Mitsubishi','Opel','Scoda','Mini','Ferrari','Lamborghini','Jaguar','Suzuki', 'Ibiza', 'Haval','GMC']

def initialize_components():
    """!Initializes necessary components for object detection.

    @return the frame's content, frame_number ,and the results of the general features (car/truck, lp, and brand) model
    """
    model = YOLO(vehicle_det_config['general_features_model'])
    cap = cv2.VideoCapture(0)  
    frame_number = 0
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()  # Create background subtractor object
    return model, cap, frame_number, bg_subtractor

def object_detection_loop(model, cap):
    """!Perform object detection on video frames.

    @param the yolo model that identifies the general features (car/truck, lp, and brand) and cap.

    @return the frame's content, frame_number ,and the results of the general features model
    """
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            raise Exception("Error: Frame could not be read.")

        frame_number += 1
        results = model.predict(frame)
        return frame, results, frame_number

def filter_process_objects(results):
    """!Filters and processes detected objects.

    @param the results of the yolo general features model.

    @return detected objects stored in a list in the following order ['car'/'truck', 'LP', 'brand'].
    """
    for result in results:
        class_ids=[]
        confidences=[]
        for result in results:
            boxes = result.boxes.cpu().numpy()
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)
            class_ids_array = np.concatenate(class_ids)
            confidences_array = np.concatenate(confidences)
            desired_class_ids_array = class_ids_array.astype(int).tolist()
            desired_conf_array = confidences_array.astype(float).tolist()
            result_dict = {}
            car_found = False
            LP_found = False
            Brand_found = False

            for i in range(len(desired_class_ids_array)):
                if desired_conf_array[i] > 0.3:
                    result_dict[desired_class_ids_array[i]] = desired_conf_array[i]

            result_dict = dict(sorted(result_dict.items()))

            if len(result_dict)>3: 
                class_ids_list = list(result_dict.keys())
                if class_ids_list[0]=='0' and class_ids_list[1]=='1': # if the model is unable to decide whether it's a car or a truck, the one with the higher confidence is gonna be kept
                    if result_dict['0']> result_dict['1']:
                        del result_dict['1']
                    else:
                        del result_dict['0']
                class_ids_list1 = list(result_dict.keys())
                while len(result_dict)>3:  # choose the brand with the higher confidence
                    if result_dict[class_ids_list1[-1]]>result_dict[class_ids_list1[-2]]:
                        del result_dict[class_ids_list1[-2]]
                    else:
                        del result_dict[class_ids_list1[-1]]
                    class_ids_list1 = list(result_dict.keys())

            features = list(result_dict.keys())
            final_features = []
            if len(result_dict) > 2:
                for j in result_dict.keys():
                    if j in [0, 1] and result_dict[j] > 0.1:  
                        car_found = True
                    elif j == 2 and result_dict[j] > 0.5:
                        LP_found = True
                    elif 3 <= j <= 35 and result_dict[j] > 0.5:
                        Brand_found = True
                if car_found and LP_found and Brand_found: # if the vehicle and the brand and the lp are detected proceed to other tests
                    for feature in features:
                        final_features.append(class_names[feature])
                    return final_features
                
def recognize_license_plate(results, frame_number):
    """!Recognizes license plate and predict nationality..

    @param the results of the general features model, frame_number.

    @return the LP's content and it's nationality. 
    """
    for result in results:
        for i in range(len(result)):
            if result[i].boxes.cpu().numpy().cls == [          2]:
                LP=""
                cropped_LP_file=""
                cropped_LP_filename = f"/home/pc/vehicle_detector/LP_cropped{frame_number}"
                cropped_LP_file = f"/home/pc/vehicle_detector/LP_cropped{frame_number}.jpg"
                result[i].save_crop("/home/pc/vehicle_detector/", cropped_LP_filename)
                LP = read_license_plate(cropped_LP_file)
                if len(LP) < 4:
                    os.remove(cropped_LP_file)
                    raise Exception("Error: LP could not be read properly, try to move a bit")


                if cropped_LP_file !="" and len(LP) > 3:
                    general_nationality = predict_generic_nationality(cropped_LP_file)
                    os.remove(cropped_LP_file)
                    return LP , general_nationality

def identify_vehicle_color(frame, contours, area_threshold, frame_number):
    """!Identify the color of the current vehicle.

    @param the frame's content, contours, area_threshold, frame_number.

    @return the vehicle's color. 
    """
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        if w * h > area_threshold:  
            screenshot_filename = f"/home/pc/vehicle_detector/_{frame_number}.jpg"
            cv2.imwrite(screenshot_filename, frame)                                           
            color = predict_car_color(screenshot_filename)
            os.remove(screenshot_filename)
            return color


def final_features_optimized(video_file):
    """!Identifies the needed features of the vehicle.
    
     @param the video's path or webcam.
    
    @return all the required features in a list in the following form ['car'/'truck', 'LP', 'brand' , 'nationality' , 'color']
    """
    # Initialize components
    model, cap, frame_number, bg_subtractor = initialize_components()

    while True: 
        # Object detection loop
        frame, results, frame_number = object_detection_loop(model, cap)
        final_features=[]
        # Filter and process detected objects
        final_features = filter_process_objects(results)
        while final_features is not None:
            # License plate recognition
            nationality = recognize_license_plate(results, frame_number)
            final_features.append(nationality[1])
            final_features[1] = nationality[0]
            area_threshold = 200
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)
            fg_mask = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)[1]
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # color identification
            color = identify_vehicle_color(frame, contours, area_threshold, frame_number)
            final_features.append(color)

            # Return final features if all required features are found
            if len(final_features) == 5:
                return final_features






with open('MQTT_config.json') as f:
    config = json.load(f)

# Global Constants for the MQTT part
broker = config['broker_address']
port = config['port']
topic = config['topic']
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 1000)}'

def connect_mqtt():
    """! connects to MQTT"""
    client = mqtt_client.Client(client_id)
    client.connect(broker, port)
    return client

def features_to_json(file_path):
    """! organizes the identified features of the vehicle into a json message.

    @param file_path path to the video or camera.

    @return a json message.
    """   
    values_list = final_features_optimized(file_path)
    uid = str(uuid.uuid4()) #! Generating a random UUID for uid
    
    json_data = {
        "activity": "Monitoring",
        "class": values_list[0],  
        "classificators": [
            {
                "brand": values_list[2],
                "class": values_list[0],  
                "color": values_list[4],
                "country": values_list[3],
                "model": "unable to identify model",
                "origin": "camera LPM",
                "registration": values_list[1],  
                "uid": uid  
            }
        ],
        "code": "1002",
        "from": "Public",
        "registration": values_list[1],  
        "to": "Parc",
        "uidpassage": uid
    }

    # Convert the dictionary to a JSON string
    json_message = json.dumps(json_data, indent=4)
    connect_mqtt().publish("features_message", json_message)

def run():
    """! sends the json message using MQTT"""
    client = connect_mqtt()
    client.loop_start()
    while True:
        features_to_json('/home/pc/Downloads/4.webm') 

run()