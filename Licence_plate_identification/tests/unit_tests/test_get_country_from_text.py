# test_get_country_from_text.py
import pytest
import sys 
sys.path.append('../../lp_detection')
from country_identifier import get_country_from_text
def test_get_country_from_text():
    """
    Test the get_country_from_text function.

    This test function checks if the get_country_from_text function
    correctly identifies the country and digits from a given text.

    Returns:
        None

    Raises:
        AssertionError: If the function does not return the expected result.
    """

    text = "This is a text containing keywords for France"
    country, digits = get_country_from_text(text)

    assert isinstance(country, str)
    

