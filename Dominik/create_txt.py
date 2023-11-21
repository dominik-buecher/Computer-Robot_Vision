def convert_annotation_format(input_file, output_file):
    with open(input_file, 'r') as input_txt, open(output_file, 'w') as output_txt:
        # Schreibe Header für die Ausgabedatei (falls erforderlich)
        output_txt.write("name class x_center y_center width height\n")

        # Lese Zeilen aus der Eingabedatei
        lines = input_txt.readlines()[1:]  # Überspringe die Header-Zeile
        for line in lines:
            data = line.strip().split(';')

            # Extrahiere Informationen
            filename = data[0]
            x1, y1, x2, y2 = map(int, data[1:5])
            width, height = map(int, data[6:8])

            # Berechne Bounding-Box Koordinaten
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            box_width = x2 - x1
            box_height = y2 - y1

            # Schreibe in die Ausgabedatei
            output_txt.write(f"dataset/positive_samples/{filename} 1 {x_center} {y_center} {box_width} {box_height}\n")


# Beispielaufruf:
input_file = 'dataset/positive_samples/gt.txt'
output_file = 'pos.txt'
convert_annotation_format(input_file, output_file)


#C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec
#C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/cascade/ -vec pos.vec -bg dataset/negative_samples/neg.txt -w 24 -h 24 -numPos 280 -numNeg 140 -numStages 12