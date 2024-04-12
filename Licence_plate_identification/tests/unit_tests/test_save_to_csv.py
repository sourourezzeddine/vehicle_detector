import os
import csv
import pytest
import sys 
sys.path.append('../../lp_detection')
from save_to_csv import save_to_csv

@pytest.fixture
def create_test_data(tmpdir):
    """
    Fixture function to create test data and a temporary CSV file path.

    This function creates test data in the form of a list of dictionaries
    and provides a path to a temporary CSV file for use in testing the save_to_csv function.

    Args:
        tmpdir: Pytest built-in fixture providing a temporary directory.

    Returns:
        tuple: A tuple containing the test data and the path to the temporary CSV file.
    """
    data = [
        {'Image': 'image1.jpg', 'Registration Number': '1234', 'Country': 'Germany'},
        {'Image': 'image2.jpg', 'Registration Number': '5678', 'Country': 'France'}
    ]
    return data, str(tmpdir.join("test_output.csv"))

def test_save_to_csv(create_test_data):
    """
    Test the save_to_csv function.

    This test function checks if the save_to_csv function correctly saves data to a CSV file.

    Args:
        create_test_data: Fixture providing the test data and the path to the temporary CSV file.

    Returns:
        None

    Raises:
        AssertionError: If the function does not save the data correctly.
    """
    data, filename = create_test_data
    save_to_csv(data, filename)

    assert os.path.exists(filename)
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        assert len(rows) == len(data)
        assert rows[0]['Image'] == data[0]['Image']
        assert rows[1]['Registration Number'] == data[1]['Registration Number']

