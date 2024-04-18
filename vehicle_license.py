import os
import cv2
import easyocr
from ultralytics import YOLO
import json

try:
    with open('main_config.json') as f:
        vehicle_det_config = json.load(f)
except :
    raise FileNotFoundError("The file 'main_config.json' was not found.")

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

def recognize_license_plate(results, frame_number):
    """!Recognizes license plate and predict nationality.

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