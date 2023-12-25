import os
import random
import shutil


def move_images_to_test_folder(input_folder, output_folder, test_percentage=0.15):
    # Erstelle den Ausgabeordner, falls er nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Durchsuche alle Unterordner im Hauptordner
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)

        # Ignoriere Dateien, falls vorhanden
        if os.path.isdir(subfolder_path):
            # Erstelle einen Testordner für den aktuellen Unterordner, falls nicht vorhanden
            test_subfolder_path = os.path.join(output_folder, subfolder)
            if not os.path.exists(test_subfolder_path):
                os.makedirs(test_subfolder_path)

            # Durchsuche alle Bilder im Unterordner
            images = os.listdir(subfolder_path)
            num_test_images = int(len(images) * test_percentage)

            # Zufällige Auswahl von Bildern für den Testordner
            test_images = random.sample(images, num_test_images)

            # Verschiebe die ausgewählten Bilder in den Testordner
            for image in test_images:
                image_path = os.path.join(subfolder_path, image)
                destination_path = os.path.join(test_subfolder_path, image)
                shutil.move(image_path, destination_path)


input_folder = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\train"
output_folder = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\test"

move_images_to_test_folder(input_folder, output_folder, test_percentage=0.5)

