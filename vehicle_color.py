"""! @brief module responsible for identifying the vehicles color."""

import torch
from torchvision import transforms
from PIL import Image
import json
import cv2
import os


try:
    with open('main_config.json') as f:
        vehicle_det_config = json.load(f)
except :
    raise FileNotFoundError("The file 'main_config.json' was not found.")

# Functions
def predict_car_color(image_to_cap):
    """predicts the vehicle color.

    @param image_to_cap  path to the vehicle image.

    @return color
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

def identify_vehicle_color(frame, contours, area_threshold, frame_number):
    """Identify the color of the current vehicle.

    @param frame image array that represents the captured frame.
    @param contours A list of contours detected in the frame using OpenCV's contour detection algorithms.
    @param area_threshold The threshold value used to filter out contours based on their area.
    @param frame_number The sequential number or index of the current frame being processed.

    @return color the vehicle's color. 
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