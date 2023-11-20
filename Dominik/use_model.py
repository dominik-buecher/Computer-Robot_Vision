#C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec
#C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/cascade_4/ -vec Dominik/pos.vec -bg dataset/negative_samples/neg.txt -precalcValBufSize 6000 -precalcIdxBufSize 6000 -w 24 -h 24 -numPos 280 -numNeg 280 -numStages 8 -maxFalseAlarmRate 0.20 -minHitRate 0.999


import cv2

def detect_and_draw_speed_sign(image_path, cascade):
    # Lade das Bild
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Erkenne Geschwindigkeitsschilder
    speed_signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    # Zeichne Bounding-Boxen um Geschwindigkeitsschilder
    for (x, y, w, h) in speed_signs:
        cv2.rectangle(image, (x-w, y-h), (x+w, y+h), (0, 255, 0), 2)

        # Extrahiere den Bereich der Bounding Box
        roi = image[y-h:y+h, x-w:x+w]

        # Zeige den Bereich der Bounding Box in einem separaten Fenster an
        #cv2.imshow('Region of Interest', roi)
        #cv2.waitKey(0)

    # Zeige das Ergebnisbild mit Bounding-Boxen an
    cv2.imshow('Detected Speed Signs', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Beispielaufruf:
cascade_speedsign = cv2.CascadeClassifier('Dominik/cascade_4/cascade.xml')
image_path = r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\Computer-Robot_Vision\dataset\positive_samples\00146.jpg'
detect_and_draw_speed_sign(image_path, cascade_speedsign)
