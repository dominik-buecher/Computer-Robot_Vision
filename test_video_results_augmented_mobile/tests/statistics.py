import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("Aktuelles Verzeichnis:", os.getcwd())


def read_labels(file_path, has_header=True):
    column_names = ['frame'] + [f'label{i}' for i in range(1, 100)]
    if file_path.endswith('.csv'):
        if has_header:
            df = pd.read_csv(file_path, names=column_names, header=0)
        else:
            df = pd.read_csv(file_path,names=column_names, header=None)
            # df = pd.read_csv(file_path, header=None)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Dateiformat wird nicht unterstützt.")

    labels = []
    for _, row in df.iterrows():
        # Sammelt alle nicht-leeren Labels, die nicht "no_sign" oder "no_speed_sign" sind
        significant_labels = set(label for label in row[1:] if not pd.isna(label) and label not in ['no_sign', 'no_speed_sign'])

        # Wenn es signifikante Labels gibt, verwenden Sie diese, sonst "no_sign"
        if significant_labels:
            labels.append(list(significant_labels))
        else:
            labels.append(['no_sign'])

    return labels



def create_confusion_matrix(actual_labels, predicted_labels, class_labels):
    num_classes = len(class_labels)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    label_to_index = {label: index for index, label in enumerate(class_labels)}

    for actual, predicted in zip(actual_labels, predicted_labels):
        for act_label in actual:
            act_index = label_to_index[act_label]
            for pred_label in predicted:
                if pred_label in label_to_index:  # Ensure the predicted label is in the class labels
                    pred_index = label_to_index[pred_label]
                    matrix[act_index, pred_index] += 1

    return matrix

def display_matrix(matrix, class_labels, video_id, total=False, show=True):
    # Save a copy of the original matrix for annotations
    annot_matrix = np.copy(matrix)

    # Transform the data
    matrix = np.log1p(matrix)

    # Create the heatmap with a color bar and annotations
    if total:
        sns.heatmap(matrix, annot=annot_matrix, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, cbar=True)
        plt.title('Logarithmische Gesamtkonfusionsmatrix')
    else:
        sns.heatmap(matrix, annot=annot_matrix, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, cbar=True)
        plt.title('Logarithmische Konfusionsmatrix - ' + video_id)
    plt.ylabel('Tatsächliche Klasse')
    plt.xlabel('Vorhergesagte Klasse')
    if show == True:
        plt.show()


def calculate_and_display_metrics(confusion_matrix, class_labels):
    num_classes = confusion_matrix.shape[0]
    total_correct = np.trace(confusion_matrix)
    total_predictions = np.sum(confusion_matrix)

    print("Metriken für jede Klasse:")
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = total_predictions - TP - FP - FN

        Precision = TP / (TP + FP) if TP + FP != 0 else 0
        Recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_Score = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0

        print(f"Klasse {class_labels[i]}:")
        print(f"    True Positives (TP): {TP}")
        print(f"    False Positives (FP): {FP}")
        print(f"    False Negatives (FN): {FN}")
        print(f"    True Negatives (TN): {TN}")
        print(f"    Precision: {Precision:.2f}")
        print(f"    Recall: {Recall:.2f}")
        print(f"    F1-Score: {F1_Score:.2f}")
        print()

    # Berechnung der Gesamtgenauigkeit
    accuracy = total_correct / total_predictions
    print(f"Gesamtgenauigkeit (Accuracy): {accuracy:.2f}")

def create_confusion_matrix_for_all_videos(video_numbers, class_labels):
    # Basispfad
    base_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_video_results_augmented_mobile\tests"
    # Initialisiere eine leere Matrix für die Gesamtkonfusionsmatrix
    total_confusion_matrix = None

    # Durchlaufe alle Verzeichnisse im Basispfad
    for video_number in os.listdir(base_path):
        # Pfad zur Konfusionsmatrix-Datei
        file_path = os.path.join(base_path, video_number, f"{video_number}_confusion_matrix.npy")

        # Überprüfe, ob die Datei existiert
        if os.path.isfile(file_path):
            # Lade die Konfusionsmatrix
            confusion_matrix = np.load(file_path)

            # Füge die Konfusionsmatrix zur Gesamtkonfusionsmatrix hinzu
            if total_confusion_matrix is None:
                total_confusion_matrix = confusion_matrix
            else:
                total_confusion_matrix += confusion_matrix

    calculate_and_display_metrics(total_confusion_matrix, class_labels)
    display_matrix(total_confusion_matrix, class_labels, video_number)
    return total_confusion_matrix


def display_confusion_matrices_for_all_videos(video_numbers, class_labels, total_confusion_matrix=None):
    # Basispfad
    base_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_video_results_augmented_mobile\tests"

    # Durchlaufe alle Videos
    for video_number in video_numbers:
        file_path = os.path.join(base_path, video_number, f"{video_number}_confusion_matrix.npy")

        if os.path.isfile(file_path):
            confusion_matrix = np.load(file_path)

            # Erstelle ein neues Fenster für jede Konfusionsmatrix
            plt.figure(figsize=(10, 8))
            display_matrix(confusion_matrix, class_labels, video_number, False, False)

    if total_confusion_matrix is not None:
        # Erstelle ein neues Fenster für die Gesamtkonfusionsmatrix
        plt.figure(figsize=(10, 8))
        display_matrix(total_confusion_matrix, class_labels, None, True, False)

    # Zeige alle Fenster an
    plt.show()

def main():
    # video_name = 'GX010108_d'
    # actual_file = fr"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_video_results_augmented_mobile\tests\{video_name}\{video_name}_labels.xlsx"
    # predicted_file = fr"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_video_results_augmented_mobile\tests\{video_name}\predicted_labels_no_audio_{video_name}.csv"
    # confusion_matrix_path = fr"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_video_results_augmented_mobile\tests\{video_name}\{video_name}_confusion_matrix.npy"

    class_labels = ['end_speed', 'no_sign', 'no_speed_sign', 'speed_100', 'speed_120',
                    'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

    actual_labels = read_labels(actual_file)

    predicted_labels = read_labels(predicted_file, False)

    matrix = create_confusion_matrix(actual_labels, predicted_labels, class_labels)
    np.save(confusion_matrix_path, matrix)

    # calculate_and_display_metrics(matrix, class_labels)
    # display_matrix(matrix, class_labels, video_name)

    video_numbers = [
    "GX010093", "GX010098", "GX010103",
    "GX010094", "GX010099", "GX010105",
    "GX010095", "GX010100", "GX010106",
    "GX010096", "GX010101", "GX010107_d",
    "GX010097", "GX010102", "GX010108_d"
    ]

    total_confusion_matrix_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_video_results_augmented_mobile\tests\total_confusion_matrix.npy"

    #total_confusion_matrix = create_confusion_matrix_for_all_videos(video_numbers, class_labels)
    #np.save(total_confusion_matrix_path, total_confusion_matrix)
    total_confusion_matrix =  np.load(total_confusion_matrix_path)

    display_confusion_matrices_for_all_videos(video_numbers, class_labels, total_confusion_matrix)

main()
