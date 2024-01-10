import os
from PIL import Image

def convert_ppm_to_jpg(parent_folder_path):
    # Iterate over all subfolders in the parent folder
    for folder_name in os.listdir(parent_folder_path):
        print(f"Converting images in folder '{folder_name}'")
        folder_path = os.path.join(parent_folder_path, folder_name)

        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        # Create a new folder for the converted jpg files
        new_folder_path = os.path.join(parent_folder_path, folder_name + "_jpg")
        os.makedirs(new_folder_path, exist_ok=True)

        # Convert ppm files to jpg
        for filename in os.listdir(folder_path):
            if filename.endswith(".ppm"):
                ppm_path = os.path.join(folder_path, filename)
                jpg_path = os.path.join(new_folder_path, os.path.splitext(filename)[0] + ".jpg")
                with Image.open(ppm_path) as img:
                    img.save(jpg_path, "JPEG")

# Example usage
folder_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision\datasets\sign_classification\GTSRB\Train\Images_ppm"
convert_ppm_to_jpg(folder_path)
