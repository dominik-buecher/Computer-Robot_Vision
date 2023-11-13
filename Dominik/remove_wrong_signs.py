# Dateipfad zur Textdatei
dateipfad = r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\dataset\GTSDB\Test\Images_jpg\gt.txt'


# Liste zum Speichern der Zeilen, die beibehalten werden sollen
beibehalten = []
counter = 0

# Datei öffnen und Zeilen verarbeiten
with open(dateipfad, 'r') as datei:
    for zeile in datei:
        # Die letzte Zahl in der Zeile extrahieren
        letzte_zahl = int(zeile.strip().split(';')[-1])

        # Überprüfen, ob die Bedingungen erfüllt sind
        if letzte_zahl < 9 or letzte_zahl == 32:
            beibehalten.append(zeile)
            counter = counter + 1

# Datei mit den beibehaltenen Zeilen überschreiben
with open(dateipfad, 'w') as datei:
    datei.writelines(beibehalten)

print("Analyse abgeschlossen. Überprüfte Zeilen wurden beibehalten, andere wurden entfernt.")
print("counter: ", counter)
