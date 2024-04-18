import main
import json


try:
   with open('test_config.json') as f:
        test_config = json.load(f)
except :
    raise FileNotFoundError("The file 'MQTT_config.json' was not found.")

test_image = test_config['test_image_color']
def test_predict_car_color():
    expected_colors = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']
    color = main.predict_car_color(test_image)
    assert  color in expected_colors

im = test_config['test_image_lp']
def test_read_license_plate():
    text = main.read_license_plate(im)
    assert text == 'B 58 BPS'

def test_predict_generic_nationality():
    nationality = main.predict_generic_nationality (im)
    assert nationality == 'europe'

