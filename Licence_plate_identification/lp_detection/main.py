# Imports
import folder_parser
import text_extractor
import country_identifier
from save_to_csv import save_to_csv
# @mainpage
# @section description_main Description
# This Python script leverages Optical Character Recognition (OCR) to extract text from images containing car license plates. The extracted text is then used to determine the nationality of the car based on the license plate information. The results are saved in a CSV (Comma Separated Values) file for further analysis or integration with other systems.
def main():
    """
    Main function to process images, extract text, identify countries, and save results to a CSV file.
    
    This function performs the following steps:
    1. Analyzes a folder containing images.
    2. Extracts text from images using OCR.
    3. Identifies the country from the extracted text.
    4. Prints information about each image, including the registration number and country.
    5. Saves the output data to a CSV file.
    """
    
    folder_path = "../images"
    image_paths = folder_parser.analyze_folder(folder_path)
    
    extracted_texts = text_extractor.extract_text_from_images(image_paths)
    
    output_data = []
    for text, image_path in zip(extracted_texts, image_paths):
        try:
            country = country_identifier.get_country_from_text(text)
            output_data.append({'Image': image_path, 'Registration Number': text, 'Country': country})
            print("Image:", image_path)
            print("Registration number:", text)
            print("Country:", country)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    save_to_csv(output_data, '../output/output.csv')

if __name__ == "__main__":
    main()

