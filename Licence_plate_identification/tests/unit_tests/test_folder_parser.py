import os
import pytest
import sys 
sys.path.append('../../lp_detection')
from folder_parser import analyze_folder

@pytest.fixture
def create_test_files(tmpdir):
    """
    Fixture function to create test files in a temporary directory.

    This function creates test files with various extensions in a temporary directory
    for use in testing the analyze_folder function.

    Args:
        tmpdir: Pytest built-in fixture providing a temporary directory.

    Returns:
        str: Path to the created temporary directory.

    """
    folder_path = tmpdir.mkdir("test_folder")
    test_files = ["test1.jpg", "test2.jpeg", "test3.png", "test4.bmp"]
    for file in test_files:
        file_path = os.path.join(folder_path, file)
        open(file_path, 'a').close()
    return str(folder_path)

def test_analyze_folder(create_test_files):
    """
    Test the analyze_folder function.

    This test function checks if the analyze_folder function correctly analyzes a folder
    and returns a list of paths to the files within that folder.

    Args:
        create_test_files: Fixture providing the path to the temporary directory containing test files.

    Returns:
        None

    Raises:
        AssertionError: If the function does not return the expected result.

    """
    folder_path = create_test_files
    expected_files = [os.path.join(folder_path, "test1.jpg"),
                      os.path.join(folder_path, "test2.jpeg"),
                      os.path.join(folder_path, "test3.png"),
                      os.path.join(folder_path, "test4.bmp")]
    # Ensure that the output of analyze_folder matches the expected_files list
    assert sorted(analyze_folder(folder_path)) == sorted(expected_files)

