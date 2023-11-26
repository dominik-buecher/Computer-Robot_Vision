import shutil
import os

def kopiere_bilder(input_txt, input_bilder_ordner, output_bilder_ordner):
    # Erstellen Sie den Ausgabeordner, wenn er nicht existiert
    if not os.path.exists(output_bilder_ordner):
        os.makedirs(output_bilder_ordner)

    # Verfolgen Sie die bereits kopierten Bilder
    kopierte_bilder = set()

    with open(input_txt, 'r') as file:
        for line in file:
            # Splitte die Zeile und extrahiere den Bildnamen
            parts = line.strip().split(',')
            bild_name = parts[0]

            # Kopiere das Bild, wenn es noch nicht kopiert wurde
            if bild_name not in kopierte_bilder:
                bild_pfad = os.path.join(input_bilder_ordner, bild_name)
                shutil.copy(bild_pfad, output_bilder_ordner)
                kopierte_bilder.add(bild_name)

if __name__ == "__main__":
    input_txt_datei = r"videos_24_11_2023\combined\combined.txt"
    input_bilder_ordner = r"videos_24_11_2023\combined"
    output_bilder_ordner = r"videos_24_11_2023\new"

    kopiere_bilder(input_txt_datei, input_bilder_ordner, output_bilder_ordner)
