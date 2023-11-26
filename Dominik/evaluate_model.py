import cv2
import os

def process_images_from_file(file_path, cascade, output_folder, ground_truth_file):
    with open(ground_truth_file, 'r') as gt_file:
        gt_lines = gt_file.readlines()

    true_positives = 0
    false_positives = 0
    actual_positives = len(gt_lines)

    for line in gt_lines:
        values = line.strip().split(' ')
        image_name = values[0]
        bbox_gt = list(map(int, values[1:]))

        image_path = os.path.join('dataset/positive_samples/', image_name)
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Erkenne Geschwindigkeitsschilder
        speed_signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

        # Zeichne die tatsächliche Bounding-Box
        cv2.rectangle(image, (bbox_gt[0], bbox_gt[1]), (bbox_gt[2], bbox_gt[3]), (255, 0, 0), 2)

        # Vergleiche mit Ground Truth
        for (cx, cy, w, h) in speed_signs:
            # Berechne die Koordinaten der vom Modell erkannten Bounding-Box
            x_detected = int(cx - w / 2)
            y_detected = int(cy - h / 2)
            bbox_detected = [x_detected, y_detected, x_detected + w, y_detected + h]

            intersection = calculate_intersection(bbox_detected, bbox_gt)
            union = calculate_union(bbox_detected, bbox_gt)
            iou = intersection / union

            # Zeichne die vom Modell erkannte Bounding-Box
            cv2.rectangle(image, (bbox_detected[0], bbox_detected[1]), (bbox_detected[2], bbox_detected[3]), (0, 255, 0), 2)

            if iou > 0.5:  # Schwellenwert für eine korrekte Erkennung
                true_positives += 1
            else:
                false_positives += 1

        # Speichere das Bild mit eingezeichneten Bounding-Boxen im Ausgabeordner
        output_path = os.path.join(output_folder, f"output_{image_name}")
        #cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Berechne Precision und Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0

    print(f'Precision: {precision}, Recall: {recall}')


# Rest des Codes bleibt unverändert
def calculate_intersection(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1[:4]  # Nur die ersten vier Werte werden verwendet
    x3, y3, x4, y4 = bbox2[:4]  # Nur die ersten vier Werte werden verwendet

    x_intersection = max(0, min(x2, x4) - max(x1, x3))
    y_intersection = max(0, min(y2, y4) - max(y1, y3))
    return x_intersection * y_intersection


def calculate_union(bbox1, bbox2):
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return area_bbox1 + area_bbox2 - calculate_intersection(bbox1, bbox2)

if __name__ == "__main__":
    cascade_speedsign = cv2.CascadeClassifier('Dominik/cascade_6/cascade.xml')
    txt_file_path = 'dataset/positive_samples/gt.txt'
    output_folder_path = 'dataset/result/'
    ground_truth_file = "dataset/positive_samples/output2.txt"    #'dataset/positive_samples/ground_truth.txt'  # Pfade anpassen

    process_images_from_file(txt_file_path, cascade_speedsign, output_folder_path, ground_truth_file)
