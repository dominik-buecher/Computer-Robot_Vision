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
        folder_path = os.path.join(base_folder, f"{i:05d}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def copy_csv(input_folder, output_folder):
    # Überprüfe, ob es eine CSV-Datei im Eingabeordner gibt
    csv_file = [file for file in os.listdir(input_folder) if file.endswith('.csv')]
    if csv_file:
        csv_file_path = os.path.join(input_folder, csv_file[0])
        output_csv_path = os.path.join(output_folder, csv_file[0])
        shutil.copy(csv_file_path, output_csv_path)

if __name__ == "__main__":
    base_output_folder = r"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\Final_Training\Images_jpg"
    create_empty_folders(base_output_folder, 43)

    # For Training Images
    for i in range(43):
        input_folder_train = fr"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\Final_Training\Images_ppm\{i:05d}"
        output_folder_train = fr"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\Final_Training\Images_jpg\{i:05d}"
        copy_csv(input_folder_train, output_folder_train)
        ppm_to_jpg(input_folder_train, output_folder_train)

    # For Test Images
    input_folder_test = r"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\Final_Test\Images_ppm"
    output_folder_test = r"C:\Users\Dominik\Documents\Studium\Master\Computer_vision\Final_Test\Images_jpg"
    copy_csv(input_folder_test, output_folder_test)
    ppm_to_jpg(input_folder_test, output_folder_test)
