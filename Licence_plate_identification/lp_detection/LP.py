# Imports
import folder_parser
import text_extractor
import country_identifier

# LP function to process images, extract text, identify countries, and return results as a list of dictionaries.
def LP():
    """
    LP function to process images, extract text, identify countries, and return results as a list of dictionaries.
    
    This function performs the following steps:
    1. Analyzes a folder containing images.
    2. Extracts text from images using OCR.
    3. Identifies the country from the extracted text.
    4. Prints information about each image, including the registration number and country.
    5. Returns the output data as a list of dictionaries.
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
    
    return output_data


LP()

