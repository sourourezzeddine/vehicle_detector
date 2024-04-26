import json

def get_country_from_text(text):
    """
    Get the country name and the number of digits in the national identification number from the provided text.

    This function checks the provided text against predefined patterns, keywords, or other identifiers 
    for each country's national identification number. It returns the country name and the number of digits 
    in the national identification number if a match is found.

    :param text: The text to be analyzed.
    :type text: str
    :return: A tuple containing the country name and the number of digits in the national identification number.
             If no match is found, it returns 'Unknown' and None.
    :rtype: tuple[str, int] or tuple[str, None]
    """
    try:
        with open('countries.json', 'r') as file:
            country_patterns = json.load(file)
    except FileNotFoundError:
        print("Error: countries.json file not found.")
        return 'Unknown', None
    except json.JSONDecodeError:
        print("Error: Unable to decode JSON in countries.json.")
        return 'Unknown', None
    
    # Iterate through each country and check for matches
    for country, keywords in country_patterns.items():
        for keyword in keywords[:-1]:
            if keyword.lower() in text.lower():
                return country, keywords[-1]  # Return the country and the number of digits
    
    # If no country is matched, return 'Unknown' and None for the number of digits
    return 'Unknown', None

