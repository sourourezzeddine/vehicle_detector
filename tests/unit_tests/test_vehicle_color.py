"""! @brief Unit test for vehicle_color module"""

import pytest
import json
import sys

try:
    with open('test_config.json') as f:
        main_config = json.load(f)
except :
    raise FileNotFoundError("The file 'test_config.json' was not found.")


sys.path.append(main_config['project_directory'])
from vehicle_color import predict_car_color, identify_vehicle_color 


def test_predict_car_color():
    """Test cases for the predict_car_color function."""
    
    # Test case 1: Test with a sample image path
    image_path = main_config['test_image_color']
    assert predict_car_color(image_path) in ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']

    # Test case 2: Test with a non-existing image path
    with pytest.raises(FileNotFoundError):
        predict_car_color("non_existing_image.jpg")
