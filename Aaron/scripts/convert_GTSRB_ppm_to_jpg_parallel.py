import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def convert_and_save_parallel(input_root, output_root):
    # Erstelle den Ausgabeordner, falls er nicht existiert
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Durchlaufe alle Unterordner im Eingabeordner
    for root, dirs, files in os.walk(input_root):
        # Erstelle den entsprechenden Ausgabeunterordner
        relative_path = os.path.relpath(root, input_root)
        output_folder = os.path.join(output_root, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        # Verarbeite alle .ppm-Bilder im aktuellen Unterordner parallel
        with ProcessPoolExecutor() as executor:
            futures = []
            for file in files:
                if file.lower().endswith(".ppm"):
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.jpg")
                    futures.append(executor.submit(convert_and_save_image, input_path, output_path))

            # Warte auf den Abschluss aller Aufgaben
            for future in futures:
                future.result()

def convert_and_save_image(input_path, output_path):
    # Lade das Bild
    image = Image.open(input_path)

    # Konvertiere und speichere das Bild als .jpg
    image.convert("RGB").save(output_path, "JPEG")


input_root = r"C:\Users\aaron\Downloads\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images"
output_root = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\GTSRB_vollständig"
# Beispielaufruf

if __name__ == '__main__':
    # Your main code here
    convert_and_save_parallel(input_root, output_root)
