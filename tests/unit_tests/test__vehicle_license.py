"""! @brief Unit test for vehicle_license module"""
import pytest
import json
import sys

try:
    with open('test_config.json') as f:
        main_config = json.load(f)
except :
    raise FileNotFoundError("The file 'test_config.json' was not found.")


sys.path.append(main_config['project_directory'])

from vehicle_license import read_license_plate, predict_generic_nationality, recognize_license_plate

def test_read_license_plate():
    """Test cases for the read_license_plate function."""
    
    # Test case 1: Test with a sample license plate image
    image_path = main_config['test_image_lp']
    lp_text = read_license_plate(image_path)
    assert isinstance(lp_text, str)  # Replace "ABC123" with expected text

    # Test case 2: Test with a non-existing image path
    with pytest.raises(Exception):
        read_license_plate("non_existing_image.jpg")

def test_predict_generic_nationality():
    """Test cases for the predict_generic_nationality function."""
    
    # Test case 1: Test with a sample license plate image
    image_path = main_config['test_image_lp']
    assert len(predict_generic_nationality(image_path)) > 0
    # Test case 2: Test with a non-existing image path
    with pytest.raises(Exception):
        predict_generic_nationality("non_existing_image.jpg")




