import os
import shutil

def move_images(source_folder, destination_folder, annotation_file):
    # Erstelle den Zielordner, wenn er nicht existiert
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Lese die Annotationen aus der TXT-Datei
    with open(annotation_file, 'r') as file:
        annotations = file.readlines()

    # Verwende ein Set, um doppelte Dateinamen zu verfolgen
    processed_filenames = set()

    for annotation in annotations:
        # Trenne die Werte in der Zeile auf
        values = annotation.strip().split(';')

        # Dateiname und Koordinaten extrahieren
        filename = values[0]

        # Überprüfe, ob das Bild bereits verschoben wurde
        if filename not in processed_filenames:
            coordinates = [int(coord) for coord in values[1:]]

            # Pfade zu den Quell- und Zielbildern erstellen
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            # Bild verschieben
            shutil.move(source_path, destination_path)

            # Füge den Dateinamen dem Set hinzu
            processed_filenames.add(filename)

            print(f"Verschoben: {filename}")
        else:
            print(f"Bild {filename} wurde bereits verschoben.")


if __name__ == "__main__":
    # Pfade anpassen
    source_folder = r"C:\Users\akiani\Documents\Studium\dataset_alles\dataset\GTSDB\Test\Images_jpg"
    destination_folder = r"C:\Users\akiani\Documents\Studium\dataset\positive_samples"
    annotation_file = r"C:\Users\akiani\Documents\Studium\dataset_alles\dataset\GTSDB\Test\Images_jpg\gt.txt"

    move_images(source_folder, destination_folder, annotation_file)
