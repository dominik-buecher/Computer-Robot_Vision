import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("Aktuelles Verzeichnis:", os.getcwd())


def read_labels(file_path, has_header=True):
    column_names = ['frame'] + [f'label{i}' for i in range(1, 1000)]
    if file_path.endswith('.csv'):
        if has_header:
            df = pd.read_csv(file_path, names=column_names, header=0, dtype=str)
        else:
            df = pd.read_csv(file_path, names=column_names, header=None, dtype=str)
            # read without column names
            #df = pd.read_csv(file_path, header=None)
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

def display_matrix(matrix, class_labels, video_id, total=False, show=False, title=False):
    # Save a copy of the original matrix for annotations
    annot_matrix = np.copy(matrix)

    # Transform the data
    matrix = np.log1p(matrix)

    # Create the heatmap with a color bar and annotations
    if total:
        sns.heatmap(matrix, annot=annot_matrix, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, cbar=True, annot_kws={"size": 12}, )
        if title:
            plt.title(title, fontsize=20)
        else:
            plt.title('Gesamtkonfusionsmatrix', fontsize=20)
    else:
        sns.heatmap(matrix, annot=annot_matrix, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, cbar=True, annot_kws={"size": 12})
        plt.title('Konfusionsmatrix - ' + video_id, fontsize=20)

    # Increase xticklabels and yticklabels font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylabel('Ground Truth', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=16)
    if show == True:
        plt.show()


def calculate_and_display_metrics(confusion_matrix, class_labels, save_excel=None):
    num_classes = confusion_matrix.shape[0]
    total_correct = np.trace(confusion_matrix)
    total_predictions = np.sum(confusion_matrix)

    metrics = []

    print("Metriken für jede Klasse:")
    total_TP = total_FP = total_FN = total_TN = 0
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = total_predictions - TP - FP - FN

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

        Precision = TP / (TP + FP) if TP + FP != 0 else 0
        Recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_Score = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0

        metrics.append([class_labels[i], TP, FP, FN, TN, Precision, Recall, F1_Score])

        print(f"Klasse {class_labels[i]}:")
        print(f"    True Positives (TP): {TP}")
        print(f"    False Positives (FP): {FP}")
        print(f"    False Negatives (FN): {FN}")
        print(f"    True Negatives (TN): {TN}")
        print(f"    Precision: {Precision:.2f}")
        print(f"    Recall: {Recall:.2f}")
        print(f"    F1-Score: {F1_Score:.2f}")

    total_Precision = total_TP / (total_TP + total_FP) if total_TP + total_FP != 0 else 0
    total_Recall = total_TP / (total_TP + total_FN) if total_TP + total_FN != 0 else 0
    total_F1_Score = 2 * (total_Precision * total_Recall) / (total_Precision + total_Recall) if total_Precision + total_Recall != 0 else 0

    metrics.append(['Gesamt', total_TP, total_FP, total_FN, total_TN, total_Precision, total_Recall, total_F1_Score])

    df = pd.DataFrame(metrics, columns=['Klasse', 'True Positives', 'False Positives', 'False Negatives', 'True Negatives', 'Precision', 'Recall', 'F1-Score'])
    print(df)

    # Berechnung der Gesamtgenauigkeit
    accuracy = total_correct / total_predictions
    print(f"Gesamtgenauigkeit (Accuracy): {accuracy:.2f}")

    # Speichern des DataFrames in einer Excel- oder CSV-Datei, wenn ein Pfad angegeben wurde
    if save_excel:
        if save_excel.endswith('.xlsx'):
            df.to_excel(save_excel, index=False)
        elif save_excel.endswith('.csv'):
            df.to_csv(save_excel, index=False)
        else:
            print("Unbekanntes Dateiformat. Bitte geben Sie einen Pfad mit der Endung .xlsx oder .csv an.")

def create_confusion_matrix_for_all_videos(base_path, class_labels):

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

    display_matrix(total_confusion_matrix, class_labels, video_number)
    return total_confusion_matrix


def display_confusion_matrices_for_all_videos(base_path, video_numbers, class_labels, total_confusion_matrix=None):

    # Durchlaufe alle Videos
    for video_number in video_numbers:
        file_path = os.path.join(base_path, video_number, f"{video_number}_confusion_matrix.npy")

        if os.path.isfile(file_path):
            matrix = np.load(file_path)

            # Erstelle ein neues Fenster für jede Konfusionsmatrix
            plt.figure(figsize=(10, 8))
            # Save a copy of the original matrix for annotations
            annot_matrix = np.copy(matrix)

            # Transform the data
            matrix = np.log1p(matrix)
            sns.heatmap(matrix, annot=annot_matrix, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, cbar=True)
            plt.title('Konfusionsmatrix - ' + video_number)
            plt.ylabel('Tatsächliche Klasse')
            plt.xlabel('Vorhergesagte Klasse')


    if total_confusion_matrix is not None:
        # Erstelle ein neues Fenster für die Gesamtkonfusionsmatrix
        plt.figure(figsize=(10, 8))
        display_matrix(total_confusion_matrix, class_labels, None, True, False)

    # Zeige alle Fenster an
    plt.show()

def display_confusion_matrices(titles, confusion_matrices, class_labels):

    # Durchlaufe alle Videos
    for i, matrix in enumerate(confusion_matrices):
        # Erstelle ein neues Fenster für jede Konfusionsmatrix
        plt.figure(figsize=(10, 8))
        display_matrix(matrix, class_labels, titles[i], False, False)

    # Zeige alle Fenster an
    plt.show()

def main():
    class_labels_less_classes = ['end_speed', 'no_sign', 'speed_100', 'speed_120',
                'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

    video_numbers = [
    "GX010093",
    "GX010098", "GX010103",
    "GX010094", "GX010099", "GX010105",
    "GX010095",
      "GX010100", "GX010106",
    "GX010096", "GX010101", "GX010107_d",
    "GX010097", "GX010102", "GX010108_d"
    ]
    conf_matrix = None

    test_paths = ["own_deeper", "own_shallow",
                  "MobileNet", "MobileNet100k", "MobileNet60k",
                  "EfficientNet100k", "EfficientNet60k"]

    # for test_path in test_paths:
        # print("\n\nCreating confusion matrices for ", test_path)
        # for video_id in video_numbers:
        #     print("Creating confusion matrix for video", video_id)
        #     actual_file = fr"test_video_results{test_path}\tests\{video_id}\{video_id}_labels.xlsx"
        #     predicted_file = fr"test_video_results{test_path}\tests\{video_id}\predicted_labels_{video_id}.csv"
        #     confusion_matrix_path = fr"test_video_results{test_path}\tests\{video_id}\{video_id}_confusion_matrix.npy"
        #     actual_labels = read_labels(actual_file)
        #     predicted_labels = read_labels(predicted_file, False)

        #     conf_matrix = create_confusion_matrix(actual_labels, predicted_labels, class_labels_less_classes)
        #     np.save(confusion_matrix_path, conf_matrix)
        # base_path = rf"test_video_results_{test_path}\tests"
        # conf_all = create_confusion_matrix_for_all_videos(base_path, class_labels_less_classes)
        # np.save(fr"test_video_results_{test_path}\tests\total_confusion_matrix.npy", conf_all)

    for test_path in test_paths:
        conf_all = np.load(fr"test_video_results_{test_path}\tests\total_confusion_matrix.npy")
        plt.figure(figsize=(10, 8))
        display_matrix(conf_all, class_labels_less_classes, None, total=True, show=False, title=test_path)

        # excel_path = fr"test_video_results_{test_path}\tests\metrics.xlsx"
        # calculate_and_display_metrics(conf_all, class_labels_less_classes, excel_path)

        # Zeige alle Fenster an
    plt.show()

    # base_path = r"test_video_results_own_deeper\tests"
    # np.save(r"test_video_results_own_deeper\tests\total_confusion_matrix.npy", conf_all)

    # base_path = r"test_video_results_MobileNet100k\tests"
    # display_confusion_matrices_for_all_videos(base_path, video_numbers, class_labels_less_classes, total_confusion_matrix=None)




main()
