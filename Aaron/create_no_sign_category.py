import os
import shutil
import random

# Pfad zu den Originalbildern
original_path = r'C:\Users\akiani\Documents\Studium\dataset_alles\dataset\GTSRB\Train\Images_jpg'

# Pfad zum Zielordner
original_target_path = r'C:\Users\akiani\Documents\Studium\Computer-Robot_Vision\datasets\sign_localization\sign_classification\GTSRB\Train\Images_jpg\no_speed'

# Unterordner, die ausgeschlossen werden sollen
excluded_folders = set(['0000{}'.format(i) for i in range(1, 9)] + ['00032'])

# Zielanzahl der Bilder
target_count = 5000

# Zufällige Auswahl der Bilder
selected_images = []

for folder_name in os.listdir(original_path):
    folder_path = os.path.join(original_path, folder_name)

    if os.path.isdir(folder_path) and folder_name not in excluded_folders:
        images = os.listdir(folder_path)
        selected_images.extend([(folder_name, img) for img in images])


random.shuffle(selected_images)
selected_images = selected_images[:target_count]

os.makedirs(original_target_path, exist_ok=True)

for folder_name, img_name in selected_images:
    source_path = os.path.join(original_path, folder_name, img_name)
    target_path = os.path.join(original_target_path, img_name)
    shutil.copy(source_path, target_path)

print(f'{len(selected_images)} Bilder wurden erfolgreich ausgewählt und in {target_path} gespeichert.')
