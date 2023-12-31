import os
from PIL import Image

def convert_png_to_jpg(png_file, jpg_file):
    with Image.open(png_file) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(jpg_file)

def rename_and_convert_images(folder_path):
    # Zähler für die Benennung der Dateien
    counter = 1

    # Liste aller .jpg und .png Dateien im Ordner erstellen
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]

    # Sortieren der Liste, um die Umbenennung in einer konsistenten Reihenfolge durchzuführen
    files.sort()

    # Durchlaufen aller Dateien im angegebenen Ordner
    for filename in files:
        # Vollständiger Pfad der aktuellen Datei
        file_path = os.path.join(folder_path, filename)

        # Neuer Dateiname für die .jpg Datei
        new_filename = f"aug_img_{counter}.jpg"
        new_file_path = os.path.join(folder_path, new_filename)

        # Prüfen, ob es sich um eine .png Datei handelt
        if filename.lower().endswith('.png'):
            # .png in .jpg konvertieren
            convert_png_to_jpg(file_path, new_file_path)
            # Ursprüngliche .png Datei löschen
            os.remove(file_path)
        else:
            # Datei umbenennen, vorhandene Dateien werden überschrieben
            os.rename(file_path, new_file_path)

        #print(f"Datei {filename} wurde zu {new_filename} umbenannt oder konvertiert")
        counter += 1

    print("Alle Dateien wurden umbenannt oder konvertiert.")


def make_images_square(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(folder_path, filename)

            with Image.open(file_path) as img:
                width, height = img.size

                # Ermitteln des kürzeren Maßes
                new_size = min(width, height)

                # Berechnen des Ausschnitts
                left = (width - new_size)/2
                top = (height - new_size)/2
                right = (width + new_size)/2
                bottom = (height + new_size)/2

                # Zuschneiden des Bildes
                img_cropped = img.crop((left, top, right, bottom))

                # Speichern des zugeschnittenen Bildes
                img_cropped.save(file_path)

                print(f"Das Bild {filename} wurde auf eine quadratische Größe von {new_size}x{new_size} zugeschnitten.")

    print("Alle Bilder wurden bearbeitet.")



def find_problematic_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            try:
                img = Image.open(os.path.join(directory, filename))
                img.verify()  # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)  # print out the names of corrupt files


input_folder = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\backup\train_more\train"

# Pfad zum Ordner mit den Bildern angeben
folder_path = 'Pfad/zu/Ihrem/Ordner'
#make_images_square(input_folder)



for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        print("Checking images in ", subfolder_path)

        # Pfad zum Ordner mit den Bildern angeben
        find_problematic_images(subfolder_path)