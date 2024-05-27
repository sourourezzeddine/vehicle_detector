# vehicle_detector
A Python codebase that detects a vehicle entering a parking lot, extracts its type (car/truck), color, brand, license plate, and nationality.

## Table_of_content

- [Description](##Description)
- [Prerequisites](##prerequisites)
- [Usage](##Usage)
- [Project Structure](##Project_Structure)
- [Data](##Data)
- [Model Architecture](## Model Architecture)
- [Acknowledgements](##Acknowledgements)
- [To Do List](##ToDo_List)

## Description
The Vehicle Detector project is designed to automate the detection and classification of vehicles entering a parking lot. It identifies the vehicle type (car/truck), color, brand, license plate, and nationality.

## Prerequisites
Ensure you have the following libraries installed. You can use the provided `requirements.txt` file to install them:

```bash
pip install -r requirements.txt
```

## Usage
To run the project, ensure all configuration files are properly set up.
Customize the paths in the following configuration files included in the repository:
    `main_config.json`
    `MQTT_config.json`
    `test_config.json`
You can then execute the whole script as: 
```bash
python3 main.py
```
## Project_Structure
vehicle_detector/
├── doc.dox
├── Flowchart_final.png
├── main_config.json
├── main.py
├── models
│   ├── best.pt
│   ├── europe_nationality_lp.pt
│   ├── final_model_85.t
│   ├── models_classes_info
│   └── nationality_generic.pt
├── MQTT_config.json
├── README.md
├── requirements.txt
├── test_config.json
├── tests
│   ├── integration_test
│   │   ├── __pycache__
│   │   │   └── test_integration.cpython-38-pytest-8.1.1.pyc
│   │   ├── test_config.json
│   │   └── test_integration.py
│   └── unit_tests
│       ├── test_config.json
│       ├── test_vehicle_color.py
│       ├── test__vehicle_license.py
│       └── test_vehicle_mqtt.py
├── vehicle_color.py
├── vehicle_features.py
├── vehicle_initialize_detection.py
├── vehicle_license.py
└── vehicle_mqtt.py
## Data
The models were trained on datasets that were manually collected and annotated in the YOLO format using [CVAT](https://cvat.org/). The dataset annotations were specifically tailored for vehicle detection tasks.

## Model Architecture
The project utilizes mainly 2 types of models models:
    [YOLOv8](https://docs.ultralytics.com) for vehicle detection and classification. YOLOv8 is chosen for its efficiency in real-time applications.
    A pretrained model for vehicle color recognition, sourced from [Vehicle-Make-Color-Recognition](https://github.com/nikalosa/Vehicle-Make-Color-Recognition).

## Acknowledgements
Special thanks to [Vehicle-Make-Color-Recognition](https://github.com/nikalosa/Vehicle-Make-Color-Recognition) for their pretrained model.

## ToDo List
- [ ] add a dashboard that displays mqtt sub data.
