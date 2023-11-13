from PIL import Image
import os
import shutil

def ppm_to_jpg(input_folder, output_folder):
    # Überprüfe, ob der Ausgabeordner existiert, andernfalls erstelle ihn
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Liste alle Dateien im Eingabeordner auf
    files = os.listdir(input_folder)

    for file in files:
        # Überprüfe, ob die Datei eine .ppm-Datei ist
        if file.endswith(".ppm"):
            # Öffne das .ppm-Bild
            ppm_path = os.path.join(input_folder, file)
            img = Image.open(ppm_path)

            # Erstelle den Dateinamen für die .jpg-Ausgabe
            jpg_path = os.path.join(output_folder, file.replace(".ppm", ".jpg"))

            # Speichere das Bild als .jpg
            img.save(jpg_path, "JPEG")


def create_empty_folders(base_folder, num_folders):
    for i in range(num_folders):
        folder_path = os.path.join(base_folder, f"{i:02d}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


if __name__ == "__main__":
    base_output_folder = r"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\Final_Training\Images_jpg"
    create_empty_folders(base_output_folder, 43)

    # Für Trainingsbilder
    for i in range(43):
        input_folder_train = fr"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\dataset\GTSDB\Evaluation\Images_ppm\{i:02d}"
        output_folder_train = fr"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\dataset\GTSDB\Evaluation\Images_jpg\{i:02d}"
        ppm_to_jpg(input_folder_train, output_folder_train)

    # Für Testbilder
    input_folder_test = r"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\dataset\GTSDB\Test\Images_ppm"
    output_folder_test = r"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\dataset\GTSDB\Test\Images_jpg"
    ppm_to_jpg(input_folder_test, output_folder_test)

