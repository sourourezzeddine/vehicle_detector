import vehicle_detector
import json

with open('test_config.json') as f:
    test_config = json.load(f)

test_image = test_config['test_image_color']
def test_predict_car_color():
    expected_colors = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']
    color = vehicle_detector.predict_car_color(test_image)
    assert  color in expected_colors

im = test_config['test_image_lp']
def test_read_license_plate():
    text = vehicle_detector.read_license_plate(im)
    assert text == 'B 58 BPS'

def test_predict_generic_nationality():
    nationality = vehicle_detector.predict_generic_nationality (im)
    assert nationality == 'europe'

