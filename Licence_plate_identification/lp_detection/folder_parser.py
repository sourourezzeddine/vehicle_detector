import os

# Functions
def analyze_folder(folder_path):
    """
    Analyze the contents of a folder and extract filenames of images.

    This function scans the specified folder for image files with common extensions (.jpg, .jpeg, .png, .bmp).
    It then returns a list of filenames of images found in the folder.

    :param folder_path: Path to the folder containing images.
    :type folder_path: str
    :return: A list of filenames of images found in the folder.
    :rtype: list[str]
    
    :raises OSError: If the specified folder does not exist or cannot be accessed.
    """
    try:
        image_filenames = []
        for filename in os.listdir(folder_path):
            # Extract base filename without extension (assumes common extensions)
            base_filename, extension = os.path.splitext(filename)
            if extension.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                image_filenames.append(os.path.join(folder_path, filename))
        return image_filenames
    except OSError as e:
        print(f"Error analyzing folder {folder_path}: {e}")
        return []

