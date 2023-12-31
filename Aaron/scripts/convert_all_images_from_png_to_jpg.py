import cv2
import os

# Pfad zum Hauptordner
main_folder_path = r'C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\backup\train'

# Zähler für die Bildnamen
image_counter = 1

# Durch alle Unterordner und Dateien in diesen Unterordnern iterieren
for root, dirs, files in os.walk(main_folder_path):
    for file in files:

        if file.endswith(".png"):
            # Pfad zur PNG-Datei
            png_file_path = os.path.join(root, file)

            # Bild mit OpenCV laden
            image = cv2.imread(png_file_path, cv2.IMREAD_UNCHANGED)

            # Formatieren des Bildnamens mit führenden Nullen
            formatted_image_name = f"{image_counter:05d}.jpg"
            jpg_file_path = os.path.join(root, formatted_image_name)

            # Bild als JPG speichern
            cv2.imwrite(jpg_file_path, image)

            # Optional: Original-PNG-Bild löschen, wenn gewünscht
            os.remove(png_file_path)

            # Zähler erhöhen
            image_counter += 1
