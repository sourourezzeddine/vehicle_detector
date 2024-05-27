"""! @brief Module that contains integration tests for validating the functionality of the Vehicle Detection System in general."""


import pytest
from unittest.mock import patch, MagicMock
import sys 
import json
import cv2
import numpy as np
import os

try:
    with open('test_config.json') as f:
        main_config = json.load(f)
except :
    raise FileNotFoundError("The file 'test_config.json' was not found.")


sys.path.append(main_config['project_directory'])
 
import main
import vehicle_initialize_detection
from vehicle_features import filter_process_objects
from vehicle_mqtt import features_to_json




# Define a fixture for a sample video file path
@pytest.fixture
def sample_video_file_path():
    """Fixture to provide the path of the sample video file for testing."""
    return main_config['video_path']

frame_number = 0

@pytest.fixture
def initialized_components(sample_video_file_path):
    """Fixture to initialize components required for testing."""
    model, cap, bg_subtractor = vehicle_initialize_detection.initialize_components(sample_video_file_path)
    yield model, cap, bg_subtractor
    # Clean up after tests
    cap.release()

def test_initialize_components(initialized_components):
    """Test the initialization of components."""
    model, cap, bg_subtractor = initialized_components
    assert model is not None
    assert isinstance(cap, cv2.VideoCapture)
    assert bg_subtractor is not None

def test_object_detection_loop(initialized_components):
    """Test the object detection loop."""
    model, cap, bg_subtractor = initialized_components
    frame_number = 0
    while True:
        frame, results, frame_number = vehicle_initialize_detection.object_detection_loop(model, cap, frame_number)
        if frame_number == 20:  # Process 10 frames for testing
            break
        assert frame is not None
        assert results is not None

def test_integration_initialize_and_loop(initialized_components):
    """Test the integration of initialization and loop."""
    model, cap, bg_subtractor = initialized_components
    frame_number = 0
    while True:
        frame, results, frame_number = vehicle_initialize_detection.object_detection_loop(model, cap, frame_number)
        if frame_number == 30:  # Process 10 frames for testing
            break
        assert frame is not None
        assert results is not None

def test_filter_process_objects(initialized_components):
    """Test the filter and process objects function."""
    model, cap, bg_subtractor = initialized_components
    frame, results, frame_numbers = vehicle_initialize_detection.object_detection_loop(model, cap, frame_number)
    processed_objects = filter_process_objects(results)
    # Assert the returned processed objects
    assert len(processed_objects) == 3

def test_final_features_optimized():
    """Test the final optimized features function."""
    features_list, returned_frame_number = main.final_features_optimized(sample_video_file_path, frame_number)
    # Assert the features list
    assert len(features_list) == 5

import os
import json
from vehicle_license import read_license_plate, predict_generic_nationality, recognize_license_plate

# Define test data directory
TEST_DATA_DIR = 'test_data'

@pytest.fixture
def sample_license_plate_image():
    """Fixture to provide the path of the sample license plate image for testing."""
    return "/home/pc/vehicle_detector/files_for_test/7.png" 

def test_read_license_plate(sample_license_plate_image):
    """Test the read license plate function."""
    lp_text = read_license_plate(sample_license_plate_image)
    assert isinstance(lp_text, str)
    assert len(lp_text) > 0  # Ensure license plate text is not empty

def test_predict_generic_nationality(sample_license_plate_image):
    """Test the predict generic nationality function."""
    country = predict_generic_nationality(sample_license_plate_image)
    assert isinstance(country, str)
    assert country in ['europe', 'america', 'qatar', 'tunisia', 'egypt', 'UAE', 'libya']

def test_recognize_license_plate(initialized_components):
    """Test the recognize license plate function."""
    # Mocking results and frame_number for testing
    model, cap, bg_subtractor = initialized_components
    frame_number = 12  # Mock frame number
    frame, results, frame_number = vehicle_initialize_detection.object_detection_loop(model, cap, frame_number)
 
    lp_content, nationality = recognize_license_plate(results, frame_number)
    assert isinstance(lp_content, str)
    assert isinstance(nationality, str)

@pytest.fixture
def sample_video_path():
    """Fixture to provide the path of the sample video for testing."""
    return "/home/pc/vehicle_detector/files_for_test/one.mp4" 

def test_features_to_json(sample_video_path, monkeypatch):
    """Test the features to JSON function."""
    # Mock MQTT connection
    class MockClient:
        def connect(self, broker, port):
            pass

        def publish(self, topic, json_message):
            assert topic == 'features_message'
            # Parse JSON message
            message = json.loads(json_message)
            assert 'activity' in message
            assert 'class' in message
            assert 'classificators' in message
            assert 'code' in message
            assert 'from' in message
            assert 'registration' in message
            assert 'to' in message
            assert 'uidpassage' in message

    # Patch connect_mqtt function to use MockClient
    def mock_connect_mqtt():
        return MockClient()

    monkeypatch.setattr('vehicle_mqtt.connect_mqtt', mock_connect_mqtt)

    # Run features_to_json with sample video path
    features_to_json(sample_video_path)
