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

# Functions
def predict_car_color(image_to_cap):
    """! predicts the vehicle color.

    @param image_to_cap  path to the vehicle image.

    @return color of the vehicle.
    """
    final_model = torch.load("/home/pc/vehicle_detector/final_model_85.t", map_location="cpu")
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
    final_model = YOLO("/home/pc/vehicle_detector/nationality_generic.pt")
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

def detect_screenshot_optimized(video_file):
    """! identifies the needed features of the vehicle.

    @param video_file path to a video or camera.

    @return list of final_features = [Vehicle type, Lp text, Brand, Country, Color].
    """    
    model = YOLO("/home/pc/vehicle_detector/best.pt")
    cap = cv2.VideoCapture(0)  
    frame_number = 0
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()  # Create background subtractor object

    while True:
        ret, frame = cap.read()
        if not ret:
            raise Exception("Error: Frame could not be read.")

        frame_number += 1
        results = model.predict(frame)

        class_ids=[]
        confidences=[]
        area_threshold = 200
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
                if class_ids_list[0]=='0' and class_ids_list[1]=='1': # if both a car and a truck are detected, keep the one with the higher confidence
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
            final_features=[]
            
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
                    fg_mask = bg_subtractor.apply(frame)  # Apply background subtraction
                    fg_mask = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)[1]  # Threshold the mask
                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
                    # License plate recognition part
                    for i in range(len(result)):
                        if result[i].boxes.cpu().numpy().cls == [          2]:
                            cropped_LP_filename = f"/home/pc/vehicle_detector/LP_cropped{frame_number}"
                            cropped_LP_file = f"/home/pc/vehicle_detector/LP_cropped{frame_number}.jpg"
                            result[i].save_crop("/home/pc/vehicle_detector/", cropped_LP_filename)
                            LP=read_license_plate(cropped_LP_file)
                            if len(LP)<4:
                                os.remove(cropped_LP_file)
                            elif len(LP)>3:
                                final_features[1]=LP
                                general_nationality=predict_generic_nationality(cropped_LP_file)
                                final_features.append(general_nationality)
                                os.remove(cropped_LP_file)
                                # Color identification part
                                if len(contours) > 0:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    x, y, w, h = cv2.boundingRect(largest_contour)
                                    if w * h > area_threshold:  
                                        screenshot_filename = f"/home/pc/vehicle_detector/_{frame_number}.jpg"
                                        cv2.imwrite(screenshot_filename, frame)                                           
                                        final_features.append(predict_car_color(screenshot_filename))
                                        os.remove(screenshot_filename)
                                        return final_features


# Global Constants for the MQTT part
broker = 'localhost'
port = 1883
topic = "features_message"
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 1000)}'

def connect_mqtt():
    """! connects to MQTT"""
    client = mqtt_client.Client(client_id)
    client.connect(broker, port)
    return client

def features_to_json(file_path):
    """! organizes the identified features of the vehicel into a json message.

    @param file_path path to the video or camera.

    @return a json message.
    """   
    values_list = detect_screenshot_optimized(file_path)
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
    result = connect_mqtt().publish("features_message", json_message)

def run():
    """! sends the json message using MQTT"""
    client = connect_mqtt()
    client.loop_start()
    while True:
        features_to_json('/home/pc/Downloads/4.webm') 

run()