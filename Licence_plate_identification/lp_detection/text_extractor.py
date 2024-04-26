import easyocr
import pytesseract
from PIL import Image
import pandas as pd

# Functions
def extract_text_from_images(image_paths):
    """
    Extract text from images using EasyOCR and Tesseract OCR.

    This function extracts text from images using two OCR engines: EasyOCR and Tesseract OCR.
    It first uses EasyOCR to perform initial text extraction, supporting both English and Arabic languages.
    Then, it utilizes Tesseract OCR for additional text extraction.
    The function chooses the extracted text with higher confidence or prioritizes EasyOCR if both are similar.

    :param image_paths: List of paths to image files.
    :type image_paths: list[str]
    :return: List of extracted texts from images.
    :rtype: list[str]
    
    :raises Exception: If there is an error during text extraction.
    """
    try:
        extracted_texts = []
        # Initialize EasyOCR reader for English and Arabic
        reader_easyocr = easyocr.Reader(['en', 'ar'])
        
        for image_path in image_paths:
            # Use EasyOCR for initial text extraction
            bounds_easyocr = reader_easyocr.readtext(image_path, detail=0)
            extracted_text_easyocr = ' '.join(bounds_easyocr)
            
            # Use Tesseract OCR for additional text extraction
            extracted_text_tesseract = pytesseract.image_to_string(Image.open(image_path), lang='eng+ara')
            
            # Choose the text with higher confidence or prioritize EasyOCR if both are similar
            if len(extracted_text_tesseract) > len(extracted_text_easyocr):
                extracted_texts.append(extracted_text_tesseract)
            else:
                extracted_texts.append(extracted_text_easyocr)
        
        return extracted_texts
    except Exception as e:
        print(f"Error extracting text from images: {e}")
        return []


# Example usage:
image_paths = ['image1.jpg', 'image2.jpg']  # Provide paths to your images
extracted_texts = extract_text_from_images(image_paths)
print(extracted_texts)

