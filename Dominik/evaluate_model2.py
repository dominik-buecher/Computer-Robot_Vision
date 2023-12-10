import cv2
import os
# image_name number_boundingbox center_x1 center_y1 width1 height1 center_x1 center_y1 width1 height1

def process_images_from_file(file_path, cascade, output_folder, ground_truth_file):
    with open(ground_truth_file, 'r') as gt_file:
        gt_lines = gt_file.readlines()

    true_positives = 0
    false_positives = 0
    actual_positives = len(gt_lines)

    for line in gt_lines:
        values = line.strip().split(' ')
        image_name = values[0]
        num_bounding_boxes = int(values[1])
        bbox_values = list(map(int, values[2:]))
        
        image_path = os.path.join('dataset/positive_samples/test/', image_name)
        image = cv2.imread(image_path)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Erkenne Geschwindigkeitsschilder
        speed_signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

        # Extrahiere tatsächliche Bounding-Boxen
        bboxes_gt = [bbox_values[i:i+4] for i in range(0, len(bbox_values), 4)]

        # Zeichne die tatsächlichen Bounding-Boxen
        for bbox in bboxes_gt:
            center_x, center_y, width, height = bbox
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = x1 + width
            y2 = y1 + height

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Vergleiche mit Ground Truth
        for (cx, cy, w, h) in speed_signs:
            for bbox_gt in bboxes_gt:
                center_x, center_y, width, height = bbox_gt
                x_gt = int(center_x - width / 2)
                y_gt = int(center_y - height / 2)
                bbox_detected = [cx, cy, w, h]

                intersection = calculate_intersection(bbox_detected, [x_gt, y_gt, x_gt + width, y_gt + height])
                union = calculate_union(bbox_detected, [x_gt, y_gt, x_gt + width, y_gt + height])
                iou = intersection / union

                # Zeichne die vom Modell erkannte Bounding-Box
                cv2.rectangle(image, (cx-(w//2), cy-(h//2)), (cx + (w//2), cy + (h//2)), (0, 255, 0), 2)

                if iou > 0.5:  # Schwellenwert für eine korrekte Erkennung
                    true_positives += 1
                else:
                    false_positives += 1

        # Speichere das Bild mit eingezeichneten Bounding-Boxen im Ausgabeordner
        output_path = os.path.join(output_folder, f"output_{image_name}")
        cv2.imwrite(output_path, image)

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
    cascade_speedsign = cv2.CascadeClassifier('Dominik/cascade_8/cascade.xml')
    if cascade_speedsign.empty():
        print("Error: Unable to load cascade classifier.")

    txt_file_path = 'dataset/positive_samples/gt.txt'
    output_folder_path = 'dataset/results4/'
    #ground_truth_file = "dataset/positive_samples/output4.txt"    #'dataset/positive_samples/ground_truth.txt'  # Pfade anpassen
    ground_truth_file = r"Dominik\annotation_files\test.txt"
    process_images_from_file(txt_file_path, cascade_speedsign, output_folder_path, ground_truth_file)
