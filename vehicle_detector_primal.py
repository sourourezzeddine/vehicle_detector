import zipfile
import requests
import cv2
import matplotlib.pyplot as plt
import glob
import random
import os
import torch
import json
import uuid
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

class_names = ['car','truck','LP','Toyota','Volkswagen','Ford','Honda','Chevrolet','Nissan','BMW','Mercedes','Audi','Tesla','Hyundai','Kia','Mazda','Fiat','Jeep','Porsche','Volvo','Land Rover','Peugeot','Renault','Citroën','Isuzu','MAN','Iveco','Mitsubishi','Opel','Scoda','Mini','Ferrari','Lamborghini','Jaguar','Suzuki', 'Ibiza', 'Haval','GMC']
basic_colors_bgr = {
    'Black': (0, 0, 0),
    'White': (255, 255, 255),
    'Gray': (128, 128, 128),
    'Red': (0, 0, 255),   # BGR format
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0),
    'Yellow': (0, 255, 255),
    'Cyan': (255, 255, 0),
    'Magenta': (255, 0, 255),
    'Orange': (0, 165, 255),  # Adjusted for BGR format
    'Pink': (203, 192, 255),  # Adjusted for BGR format
    'Brown': (42, 42, 165),   # Adjusted for BGR format
    'Gold': (0, 215, 255),    # Adjusted for BGR format
    'Silver': (192, 192, 192)
}
def remove_duplicates(file_path,conf_threshold):
    seen_classes = {}
    objects_detected = []
    with open(file_path, "r+") as f:
        lines = f.readlines()
        f.seek(0)  # Reset file pointer to the beginning
        for line in lines:
            data = line.split()  # Extract data from line
            class_id, conf = data[0], data[5]
            if class_id not in seen_classes or conf > seen_classes[class_id][1]:
                if float(conf)>conf_threshold:
                    objects_detected.append(class_id)
                    f.write(line)
                    seen_classes[class_id] = (data, conf)  # Update seen data and conf
        lines=seen_classes
        f.truncate()  # Remove any remaining data beyond the written lines
        return objects_detected 

def screenshot_to_txt(screenshot_path, txt_file): #, output_file):
    model = YOLO("/home/pc/vehicle_detector/old/best(copy).pt")
    cap = cv2.imread(screenshot_path)
    results = model.predict(cap)
    with open(txt_file, 'r+') as f:
        for result in results:
            full_path = os.path.join('/home/pc/vehicle_detector/old/', txt_file)
            result.save_txt(full_path, save_conf=True)
            return txt_file
        
m = YOLO('yolov8n-seg.pt')

def segment_car(image_path,save_dir):
    # Perform object detection on the input image
    res = m.predict(image_path)

    for r in res:
        img = np.copy(r.orig_img)
        img_name = Path(image_path).stem

        for ci, c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]
            if label == 'car' or label== 'truck':
                b_mask = np.zeros(img.shape[:2], np.uint8)
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                # -1: Isolate object with black background
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                isolated = cv2.bitwise_and(mask3ch, img)
                x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                iso_crop = isolated[y1:y2, x1:x2]
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"{img_name}_car_{ci}.jpg")
                cv2.imwrite(save_path, iso_crop)

                return save_path
def detect_car_color(car_image_path):
    car_image = cv2.imread(car_image_path)
    hsv_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 0])  # Lower HSV threshold for red
    upper_color = np.array([255, 255, 255])  # Upper HSV threshold for red
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    largest_contour_index = np.argmax(contour_areas)
    largest_contour = contours[largest_contour_index]
    mean_color = cv2.mean(car_image, mask=mask)[:3]
    closest_color = min(basic_colors_bgr, key=lambda x: np.linalg.norm(np.array(basic_colors_bgr[x]) - mean_color))

    return closest_color
def detect_and_screenshot(video_file):
    model = YOLO("/home/pc/vehicle_detector/old/best(copy).pt")
    cap = cv2.VideoCapture(video_file)
    frame_number = 0
    car_present = False  # Flag to track presence of car in previous frame
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()  # Create background subtractor object

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            results = model.predict(frame)

            # Iterate over the results to find the first matching object
            car_found = False  # Flag to track presence of car in current frame
            LP_found = False
            Brand_found = False
            area_threshold = 200
            for result in results:
                txt_file = 'yolo_output.txt'
                full_path = os.path.join('/home/pc/vehicle_detector/old/', txt_file)
                result.save_txt(full_path, save_conf=True)
                #detected_obj = remove_duplicates(('yolo_output.txt'), 0.4)
                with open('/home/pc/vehicle_detector/old/yolo_output.txt', 'r+') as f:
                    lines = f.readlines()
                    f.seek(0)  # Move pointer to the beginning of the file
                    f.truncate()  # Truncate the file content
                for line in lines:
                    class_id = int(line.split()[0])  # Extract class ID
                    vehicle_conf = float(line.split()[-1])  # Extract vehicle conf
                    if class_id in [0, 1] and vehicle_conf > 0.6:  # Check if the class ID matches 0 or 1
                        car_found = True
                    elif class_id == 2 and vehicle_conf > 0.5:
                        LP_found = True
                    elif 3 <= class_id <= 35 and vehicle_conf > 0.3:
                        Brand_found = True

                if len(lines) > 2 and car_found and LP_found and Brand_found and not car_present:
                    # Check if the car has stopped by analyzing its movement
                    fg_mask = bg_subtractor.apply(frame)  # Apply background subtraction
                    fg_mask = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)[1]  # Threshold the mask
                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
                    if len(contours) > 0:
                        # Get the bounding box of the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        if w * h > area_threshold:  # Adjust area_threshold as per your requirement
                            # Car has stopped, take a screenshot
                            screenshot_filename = f"/home/pc/vehicle_detector/old/_{frame_number}.jpg"
                            cv2.imwrite(screenshot_filename, frame)
                            print(f"Screenshot saved as {screenshot_filename}")
                            txt_file = '/home/pc/vehicle_detector/old/yolo_output2.txt'
                            detected_obj = remove_duplicates(screenshot_to_txt(screenshot_filename, txt_file),0.3)
                            if len(detected_obj) < 2:
                                os.remove(screenshot_filename)
                            else:
                                print(detected_obj)
                                
                            with open(txt_file, 'w'):
                                pass
                            
                        bg_subtractor = cv2.createBackgroundSubtractorMOG2()  # Reset background subtractor
                        return detected_obj, screenshot_filename
                car_present = car_found

    finally:
        cap.release()

def detect_and_get_class_names(video_path):
    objects = detect_and_screenshot(video_path)
    class_names_list = []
    class_names_ = []
    save_dir ='/home/pc/vehicle_detector/old/screenshot/'
    for obj in objects[0]:
        class_names_list.append(class_names[int(obj)])
    print(objects[1])
    color = detect_car_color(segment_car(objects[1],save_dir))
    class_names_list.append(color)
    return class_names_list

features = detect_and_get_class_names('/home/pc/Downloads/1.mp4')
print(features)