import os
import pytest
import sys 
from unittest.mock import MagicMock
sys.path.append('../../lp_detection')

from folder_parser import analyze_folder 
from text_extractor import extract_text_from_images
from country_identifier import get_country_from_text
from save_to_csv import save_to_csv
from main import main 

import os
import tempfile

@pytest.fixture
def create_test_images():
    """
    Fixture function to create temporary test images.

    This function creates temporary test images in a temporary directory.

    Returns:
        list: List of paths to the created test images.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_images = ['test_image1.jpg', 'test_image2.jpeg']
        for image_name in test_images:
            with open(os.path.join(temp_dir, image_name), 'w') as f:
                f.write("dummy image content")
        yield [os.path.join(temp_dir, image) for image in test_images]

def test_analyze_folder(create_test_images):
    """
    Test the analyze_folder function.

    This test function checks if the analyze_folder function correctly analyzes a folder
    and returns a list of paths to the files within that folder.

    Args:
        create_test_images: Fixture providing the paths to the temporary test images.

    Returns:
        None

    Raises:
        AssertionError: If the function does not return the expected result.
    """
    folder_path = os.path.dirname(create_test_images[0])
    result = analyze_folder(folder_path)
    assert len(result) == 2  # Assuming 2 images are created in the temporary directory
    assert all(os.path.isfile(image) for image in result)

def test_extract_text_from_images(create_test_images):
    """
    Test the extract_text_from_images function.

    This test function checks if the extract_text_from_images function correctly extracts text
    from a list of image files.

    Args:
        create_test_images: Fixture providing the paths to the temporary test images.

    Returns:
        None

    Raises:
        AssertionError: If the function does not return the expected result.
    """
    result = extract_text_from_images(create_test_images)
    assert len(result) == 0  # Assuming 2 images are created
    assert all(isinstance(text, str) for text in result)

def test_get_country_from_text():
    """
    Test the get_country_from_text function.

    This test function checks if the get_country_from_text function correctly identifies the country
    mentioned in a given text.

    Returns:
        None

    Raises:
        AssertionError: If the function does not return the expected result.
    """
    text = "This is a sample text mentioning Germany."
    country, num_digits = get_country_from_text(text)
    assert country == "Germany"
    assert num_digits == 11

def test_save_to_csv():
    """
    Test the save_to_csv function.

    This test function checks if the save_to_csv function correctly saves data to a CSV file.

    Returns:
        None

    Raises:
        AssertionError: If the function does not save the data correctly.
    """
    data = [
        {'Image': 'test_image1.jpg', 'Registration Number': '123456789', 'Country': 'Germany'},
        {'Image': 'test_image2.jpeg', 'Registration Number': '987654321', 'Country': 'France'}
    ]
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='') as temp_csv:
        filename = temp_csv.name
        save_to_csv(data, filename)
        # Check if file is created and contains correct data
        with open(filename, 'r') as f:
            csv_content = f.readlines()
            assert len(csv_content) == 3  # Header + 2 rows
            assert "test_image1.jpg,123456789,Germany\n" in csv_content
            assert "test_image2.jpeg,987654321,France\n" in csv_content

    # Clean up
    os.unlink(filename)

