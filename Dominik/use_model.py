#C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_createsamples.exe -info dataset/positive_samples/pos_new.txt -w 24 -h 24 -num 5000 -vec dataset/positive_samples/pos.vec
#C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/cascade_7/ -vec dataset/positive_samples/pos.vec -bg dataset/negative_samples/neg.txt -precalcValBufSize 10000 -precalcIdxBufSize 10000 -w 24 -h 24 -numPos 3500 -numNeg 3500 -numStages 5 -maxFalseAlarmRate 0.05 -minHitRate 0.999
#C:/Users/domin/Documents/opencv_old/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/cascade_5/ -vec dataset/positive_samples/pos.vec -bg dataset/negative_samples/neg.txt -precalcValBufSize 8000 -precalcIdxBufSize 8000 -w 24 -h 24 -numPos 3800 -numNeg 2000 -numStages 10 -maxFalseAlarmRate 0.05 -minHitRate 0.999

import cv2
import os

counter = 0  # Hier wird die Variable counter global initialisiert

def detect_and_save_speed_signs(image_path, cascade, output_folder):
    global counter  # Weise darauf hin, dass counter global verwendet wird
    # Lade das Bild
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Erkenne Geschwindigkeitsschilder
    speed_signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    # Zeichne Bounding-Boxen auf das Originalbild
    for (x, y, w, h) in speed_signs:
        cv2.rectangle(image, (x-w, y-h), (x + w, y + h), (0, 255, 0), 2)

    # Erstelle den Ausgabeordner, wenn er noch nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Speichere das Bild mit eingezeichneten Bounding-Boxen im Ausgabeordner
    output_path = os.path.join(output_folder, f"output_{counter}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    counter = counter + 1

def process_images_from_file(file_path, cascade, output_folder):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Splitte die Zeile anhand des Semikolons
        values = line.strip().split(';')

        # Extrahiere den Dateinamen und die Bounding Box-Koordinaten
        image_name = values[0]
        bbox = list(map(int, values[1:]))

        # Setze den vollständigen Pfad zum Bild
        image_path = os.path.join('dataset/positive_samples/', image_name)

        # Rufe die Funktion zur Erkennung und Zeichnung auf
        detect_and_save_speed_signs(image_path, cascade, output_folder)

if __name__ == "__main__":
    # Beispielaufruf:
    cascade_speedsign = cv2.CascadeClassifier('Dominik/cascade_6/cascade.xml')
    txt_file_path = 'dataset/positive_samples/gt.txt'
    output_folder_path = 'dataset/result/'

    process_images_from_file(txt_file_path, cascade_speedsign, output_folder_path)
