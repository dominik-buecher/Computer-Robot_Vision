## Create .vec file and train model with opencv program:
# C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_createsamples.exe -info dataset/positive_samples/train/train.txt -w 24 -h 24 -num 5000 -vec dataset/positive_samples/train/train.vec
# C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/cascade_8/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 14000 -precalcIdxBufSize 14000 -w 24 -h 24 -numPos 3500 -numNeg 7000 -numStages 5 -maxFalseAlarmRate 0.1 -minHitRate 0.999
# C:/Users/domin/Documents/opencv_old/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/cascade_5/ -vec dataset/positive_samples/pos.vec -bg dataset/negative_samples/neg.txt -precalcValBufSize 8000 -precalcIdxBufSize 8000 -w 24 -h 24 -numPos 3800 -numNeg 2000 -numStages 10 -maxFalseAlarmRate 0.05 -minHitRate 0.999 -mode ALL
#C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/cascade_10/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 14000 -precalcIdxBufSize 14000 -w 24 -h 24 -numPos 3500 -numNeg 7000 -numStages 5 -maxFalseAlarmRate 0.1 -minHitRate 0.999 -featureType LBP

# opencv_visualisation --image=/data/object.png --model=/data/model.xml --data=/data/result/

# cascade_8
# cascade_9     C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_10/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 14000 -precalcIdxBufSize 14000 -w 24 -h 24 -numPos 3500 -numNeg 7000 -numStages 5 -maxFalseAlarmRate 0.1 -minHitRate 0.999 mode ALL              *5: Precision: 0.8577981651376146, Recall: 0.8577981651376146



# cascade_13    C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_13/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 7000 -numStages 7 -maxFalseAlarmRate 0.05 -minHitRate 0.999                       5: Precision: 0.7777777777777778, Recall: 0.7659574468085106   *7: Precision: 0.8894009216589862, Recall: 0.8853211009174312   10:
# cascade_17    C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_17/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 3500 -numStages 7 -maxFalseAlarmRate 0.05 -minHitRate 0.999                      *5: Precision: 0.8073394495412844, Recall: 0.8073394495412844   *7: Precision: 0.9036697247706422, Recall: 0.9036697247706422   10:
# cascade_14    C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_14/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 1750 -numStages 7 -maxFalseAlarmRate 0.05 -minHitRate 0.999                      *5: Precision: 0.7247706422018348, Recall: 0.7247706422018348   *7: Precision: 0.8623853211009175, Recall: 0.8623853211009175   10:

# cascade_12    C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_12/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 1750 -numStages 10 -maxFalseAlarmRate 0.05 -minHitRate 0.999 -featureType LBP      5: Precision: 0.7978723404255319, Recall: 0.7978723404255319   *7: Precision: 0.8976744186046511, Recall: 0.8853211009174312   10:
# cascade_16    C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_16/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 3500 -numStages 7 -maxFalseAlarmRate 0.05 -minHitRate 0.999 -featureType LBP      5: Precision: 0.8783570300157978, Recall: 0.8449848024316109   *7: Precision: 0.9476439790575916, Recall: 0.8302752293577982   10:
# cascade_15    C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_15/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 7000 -numStages 8 -maxFalseAlarmRate 0.05 -minHitRate 0.999 -featureType LBP     *5: Precision: 0.9045226130653267, Recall: 0.8256880733944955   *7: Precision: 0.9677419354838710, Recall: 0.8256880733944955   8:



# cascade_20   C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_20/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 1750 -numStages 7 -maxFalseAlarmRate 0.01 -minHitRate 0.999 -featureType LBP       5: Precision: 0.8635658914728682, Recall: 0.8465045592705167   *7: Precision: 0.9421052631578948, Recall: 0.8211009174311926
# cascade_19   C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_19/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 3500 -numStages 7 -maxFalseAlarmRate 0.01 -minHitRate 0.999 -featureType LBP      *5: Precision: 0.9381443298969072, Recall: 0.8348623853211009   *7:
# cascade_18   C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_18/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 7000 -numStages 5 -maxFalseAlarmRate 0.01 -minHitRate 0.999 -featureType LBP      *5:


# cascade_10	C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_10/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 7000 -numStages 10 -maxFalseAlarmRate 0.1 -minHitRate 0.999 -featureType LBP      5: Precision: 0.9196428571428571, Recall: 0.7826747720364742   *7: Precision: 0.8644859813084113, Recall: 0.8486238532110092
# cascade_11    C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_11/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 1750 -numStages 12 -maxFalseAlarmRate 0.1 -minHitRate 0.999 -featureType LBP      5: Precision: 0.7082066869300911, Recall: 0.7082066869300911   7: Precision: 0.8703703703703703, Recall: 0.8571428571428571  10: Precision: 0.9270462633451957, Recall: 0.7917933130699089   *12: Precision: 0.925531914893617, Recall: 0.7981651376146789


# cascade_21   C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_21/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 1750 -numStages 5 -maxFalseAlarmRate 0.2 -minHitRate 0.999                       *3: Precision: 0.945054945054945, Recall: 0.7889908256880734
# cascade_22   C:/Users/Dominik/Documents/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data Dominik/models/cascade_22/ -vec dataset/positive_samples/train/train.vec -bg dataset/negative_samples/neg2.txt -precalcValBufSize 16000 -precalcIdxBufSize 16000 -w 24 -h 24 -numPos 3500 -numNeg 7000 -numStages 5 -maxFalseAlarmRate 0.2 -minHitRate 0.999	                      *5: Precision: 0.3394495412844037, Recall: 0.3394495412844037


import time
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def process_images_from_file(cascade, output_folder, ground_truth_file):
    with open(ground_truth_file, 'r') as gt_file:
        gt_lines = gt_file.readlines()

    true_positives = 0
    false_positives = 0
    actual_positives = 0

    for line in gt_lines:
        values = line.strip().split(' ')
        image_name = values[0]
        num_bounding_boxes = int(values[1])
        bbox_values = list(map(int, values[2:]))
        actual_positives += num_bounding_boxes
        image_path = os.path.join('dataset/positive_samples/test/', image_name)
        image = cv2.imread(image_path)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Erkenne Geschwindigkeitsschilder
        speed_signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))

        # Extrahiere tatsächliche Bounding-Boxen
        bboxes_gt = [bbox_values[i:i+4] for i in range(0, len(bbox_values), 4)]

        # Vergleiche mit Ground Truth
        for bbox_gt in bboxes_gt:
            closest_sign = None
            min_distance = float('inf')

            for (cx, cy, w, h) in speed_signs:
                distance = abs(cx - bbox_gt[0])
                if distance < min_distance:
                    min_distance = distance
                    closest_sign = (cx, cy, w, h)

            if closest_sign is not None:
                cx, cy, w, h = closest_sign
                intersection = calculate_intersection([cx, cy, w, h], bbox_gt)
                union = calculate_union([cx, cy, w, h], bbox_gt)
                iou = intersection / union

                if iou > 0.5:  # Schwellenwert für eine korrekte Erkennung
                    true_positives += 1
                else:
                    false_positives += 1



    false_negative = actual_positives - true_positives
    true_negative = 0
    # Print additional metrics (Precision, Recall, etc.)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    print(f'Precision: {precision}, Recall: {recall}')

    conf_matrix = np.array([[true_positives, false_positives],
                       [false_negative, true_negative]])

    # Plot Confusion Matrix
    labels = ['Positive', 'Negative']
    categories = ['True Positive', 'False Positive', 'False Negative', 'True Negative']

    sns.heatmap(conf_matrix, annot=np.array([[f'TP: {true_positives}', f'FP: {false_positives}'], [f'FN: {false_negative}', 'TN: Not available']]),
                fmt='', cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={'size': 14})

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Number of ground truth data: {actual_positives})')
    plt.show()

def calculate_intersection(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return x_intersection * y_intersection

def calculate_union(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    area_bbox1 = w1 * h1
    area_bbox2 = w2 * h2
    intersection = calculate_intersection(bbox1, bbox2)
    union = area_bbox1 + area_bbox2 - intersection
    return union



def test_speed(cascade):

    # Pfad zum Bild, das du klassifizieren möchtest
    image_path = r'dataset\positive_samples\test\frame_2303.jpg'  # Ersetze dies durch den tatsächlichen Pfad zu deinem Bild

    # Lade das Bild
    image = cv2.imread(image_path)

    # Konvertiere das Bild in Graustufen (Cascade-Modelle arbeiten oft besser in Graustufen)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Messen der Zeit vor der Klassifikation
    start_time = time.time()

    # Anwenden des Cascade-Modells auf das Bild
    objects = cascade.detectMultiScale(gray_image, scaleFactor=1.4, minNeighbors=5, minSize=(24, 24))

    # Messen der Zeit nach der Klassifikation
    end_time = time.time()

    # Ausgabe der erkannten Objekte und der benötigten Zeit
    print("Recognised objects:", len(objects))
    print("Time for the localization: {:.4f} Seconds".format(end_time - start_time))

    # Zeichne Rechtecke um die erkannten Objekte im Bild
    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Anzeigen des Bilds mit den erkannten Objekten
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def create_grafic():



if __name__ == "__main__":
    cascade_speedsign = cv2.CascadeClassifier('Dominik/models/cascade_15/cascade.xml')
    if cascade_speedsign.empty():
        print("Error: Unable to load cascade classifier.")

    output_folder_path = 'dataset/results4/'
    ground_truth_file = r"Dominik\annotation_files\test.txt"

    process_images_from_file(cascade_speedsign, output_folder_path, ground_truth_file)
    test_speed(cascade_speedsign)
