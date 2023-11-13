import os
from PIL import Image

# Ordnerpfad mit den JPG-Bildern
ordnerpfad = r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\dataset\GTSDB\Test\Images_jpg'

# Dateipfad zur Textdatei mit Bildinformationen
bildinfo_dateipfad = r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\dataset\GTSDB\Test\Images_jpg\gt.txt'

# Durch die Dateien im Ordner iterieren
with open(bildinfo_dateipfad, 'r') as datei:
    bildinformationen = []
    for zeile in datei:
        bildinformation_teile = zeile.strip().split(';')
        bildname = bildinformation_teile[0]
        bild_dateipfad = os.path.join(ordnerpfad, f'{bildname}')

        # Bild laden
        try:
            bild = Image.open(bild_dateipfad)

            # Größe des Bildes abrufen
            breite, höhe = bild.size

            # Bildinformationen zur Liste hinzufügen
            bildinformation_teile.extend([str(breite), str(höhe)])
            bildinformationen.append(';'.join(bildinformation_teile))

        except FileNotFoundError:
            print(f'Das Bild {bildname}.jpg wurde nicht gefunden.')

# Bildinformationen in die Ausgabedatei schreiben
with open(bildinfo_dateipfad, 'w') as ausgabe:
    ausgabe.write('\n'.join(bildinformationen))

print(f'Höhe und Breite der Bilder wurden zu {bildinfo_dateipfad} hinzugefügt.')
