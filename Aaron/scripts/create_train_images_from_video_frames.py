import os
import shutil

# Pfad zum Hauptordner, der die Unterordner enthält
main_folder_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\videos_24_11_2023"
# Zielverzeichnis, wo die neuen Ordner erstellt werden
destination_folder_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\train"

# Überprüfen, ob das Zielverzeichnis existiert, und es ggf. erstellen
if not os.path.exists(destination_folder_path):
    os.makedirs(destination_folder_path)

# Durchlaufen aller Unterordner im Hauptordner
for folder_name in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, folder_name, 'frames')
    new_folder_path = os.path.join(destination_folder_path, folder_name)

    # Überprüfen, ob der Unterordner 'frames' existiert
    if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
        # Erstellen eines neuen Ordners im Zielverzeichnis, falls er noch nicht existiert
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        # Kopieren aller Bilder aus dem 'frames'-Ordner in den neuen Ordner
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, new_folder_path)

        print(f"Bilder aus '{folder_name}/frames' wurden nach '{new_folder_path}' kopiert.")
