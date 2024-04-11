import vehicle_detector


test_image = "/home/pc/vehicle_detector/images/1.png"
def test_Predict_car_color(test_image):
    expected_colors = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']
    color = vehicle_detector.predict_car_color(test_image)
    assert  color in expected_colors

im = "/home/pc/vehicle_detector/images/7.png"
def test_read_license_plate(im):
    text = vehicle_detector.read_license_plate(im)
    assert text == 'B 58 BPS'

def test_predict_generic_nationality(im):
    nationality = vehicle_detector.predict_generic_nationality (im)
    assert nationality == 'europe'

