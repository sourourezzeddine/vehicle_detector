import csv
import os

# Functions
def save_to_csv(data, filename):
    """
    Save data to a CSV file.

    This function takes a list of dictionaries containing data and saves it to a CSV file.
    
    :param data: Data to be saved to the CSV file.
    :type data: list[dict]
    :param filename: Path to the output CSV file.
    :type filename: str
    """
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Image', 'Registration Number', 'Country']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

