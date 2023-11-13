# Funktion zum Ersetzen von ".ppm" durch ".jpg" in einer Zeile
def ersetze_ppm_durch_jpg(zeile):
    return zeile.replace(".ppm", ".jpg")

# Dateipfad zur .txt-Datei
dateipfad = r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\dataset\GTSDB\Test\Images_jpg\gt.txt'

# Liste zum Speichern der modifizierten Zeilen
modifizierte_zeilen = []

# Die Datei öffnen und Zeilen durchgehen
with open(dateipfad, 'r') as datei:
    for zeile in datei:
        modifizierte_zeilen.append(ersetze_ppm_durch_jpg(zeile))

# Die modifizierten Zeilen in die gleiche Datei schreiben
with open(dateipfad, 'w') as datei:
    for zeile in modifizierte_zeilen:
        datei.write(zeile)

print("Ersetzung abgeschlossen.")
