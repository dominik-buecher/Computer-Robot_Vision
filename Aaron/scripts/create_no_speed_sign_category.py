import os
import shutil
import random

# Pfad zu den Originalbildern
original_path = r'C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\GTSRB_no_speed'

# Pfad zum Zielordner
original_target_path = r'C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\no_speed_sign'

def copy_and_number_images(source_folder, destination_folder):
    # Create the destination folder if it does not exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Counter for image numbering
    image_number = 1

    for folder_name, subfolders, filenames in os.walk(source_folder):
        for filename in filenames:
            if filename.lower().endswith('.jpg'):
                full_file_path = os.path.join(folder_name, filename)
                destination_filename = f"no_speed_sign_{image_number}.jpg"
                destination_file_path = os.path.join(destination_folder, destination_filename)
                shutil.copy(full_file_path, destination_file_path)
                image_number += 1



def copy_random_images(source_folder, destination_folder, num_images):
    # Erstelle den Zielordner, falls er nicht existiert
    os.makedirs(destination_folder, exist_ok=True)

    # Durchsuche alle Dateien im Quellordner
    all_images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Wähle zufällig 'num_images' Bilder aus
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    # Kopiere ausgewählte Bilder in den Zielordner
    for image_name in selected_images:
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copyfile(source_path, destination_path)


# copy_and_number_images(original_path, original_target_path)

source_folder = r'C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\no_speed_sign'
destination_folder = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\train\no_speed_sign"
num_images = 5000
copy_random_images(source_folder, destination_folder, num_images)