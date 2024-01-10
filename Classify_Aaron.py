import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import utils
from tensorflow.python.keras.utils import np_utils
from keras.src.utils.np_utils import generic_utils
#from tensorflow.keras.utils import generic_utils
import shutil
import time
import os
import csv

def classify_video_batch(input_video_path, output_video_path, cnn_model_path, cascade_path):

     # Get the directory of the input video
    target_dir = os.path.dirname(output_video_path)

    # Get the video name
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # Create a CSV file name with the video name
    csv_file_name = 'predicted_labels_' + video_name + '.csv'
    # Create a CSV file in the same directory
    csv_file_path = os.path.join(target_dir, csv_file_name)

    # Create a directory for the classified_video and the CSV file
    output_path = os.path.dirname(output_video_path)
    os.makedirs(output_path, exist_ok=True)

    # Copy the Excel file with the Ground Truth Labels into the output directory and keep the file name
    ground_truth_source_path = rf"test_videos_with_labels\done\{video_name}_labels.xlsx"

    # take the path string except the last part (the file name) and copy the labels file to the output directory
    ground_truth_target_dir = os.path.dirname(output_video_path)
    shutil.copy(ground_truth_source_path, ground_truth_target_dir)

    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    get_warm = np.zeros((1, 128, 128, 3), dtype=np.uint8)

    start_time = time.time()  # Starte die Zeitmessung
    cnn_model(get_warm)
    print(f"CNN Warmup took {time.time() - start_time} seconds")

    # Öffne das Video
    cap = cv2.VideoCapture(input_video_path)

    # Erstelle einen VideoWriter für das Ergebnisvideo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # oder 'XVID' je nach Codec
    output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    if True:
        class_names = ['end_speed', 'no_sign', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']
        print("Using less classes: ", class_names)
    else:
        class_names = ['end_speed', 'no_sign', 'no_speed_sign', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']
        print("Using all classes: ", class_names)

    save_roi_path = os.path.dirname(output_video_path) + "\\rois"

    cnn_total_time = 0
    haar_total_time = 0
    cnn_counter_predictions = 0
    haar_counter_predictions = 0

    roi_count = 0
    start_time = time.time()  # Starte die Zeitmessung
    frame_number = 0


    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)

        while True:
            # Lese ein Frame aus dem Video
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Wende das Haar Cascade-Modell auf das Frame an, um Schilder zu erkennen
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            before_haar = time.time()  # Starte die Zeitmessung
            signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
            haar_total_time += (time.time() - before_haar)  # Stoppe die Zeitmessung
            haar_counter_predictions += 1

            frame_sign_rois = []
            coords = []
            for i, (x, y, w, h) in enumerate(signs):
                # Bereinige die Koordinaten der Bounding Box
                if (y - (h) < 0) or (y + (h) > 1080) or (x - (w) < 0) or x + (w) > 1920:
                    continue

                # Schneide die Bounding Box aus dem Bild aus
                sign_roi = frame[y - h:y + h, x - w:x + w]

                # Reduziere die Größe um 10% (optional, falls benötigt)
                width_reduction = int(w * 0.3)
                height_reduction = int(h * 0.3)

                sign_roi_cropped = sign_roi[height_reduction:-height_reduction, width_reduction:-width_reduction]

                # Konvertiere das ROI in RGB und skaliere es
                color_roi = cv2.cvtColor(sign_roi_cropped, cv2.COLOR_BGR2RGB)
                resized_roi = cv2.resize(color_roi, (128, 128)).astype(np.uint8)
                frame_sign_rois.append(resized_roi)

                # update the coordinates of the bounding box after cropping
                coords.append((x, y, w- width_reduction, h - height_reduction))

            predicted_classes = []
            if len(coords) > 0:
                frame_sign_rois = np.array(frame_sign_rois)
                before_pred = time.time()  # Starte die Zeitmessung
                predictions = cnn_model(frame_sign_rois)
                cnn_total_time += (time.time() - before_pred)  # Stoppe die Zeitmessung
                cnn_counter_predictions += 1

                # Schreibe das Frame in das Ausgabevideo
                for i, (x, y, w, h) in enumerate(coords):
                    # Berechne die Koordinaten der Bounding Box nach der Reduzierung der Größe

                    class_index = np.argmax(predictions[i])
                    class_prob = np.max(predictions[i])
                    class_prob = np.round(class_prob, 2)

                    x_start = x - w
                    y_start = y - h
                    x_end = x + w
                    y_end = y + h

                    # save the bounding box to the folder with predicted class
                    class_name = class_names[class_index]

                    directory = os.path.join(save_roi_path, class_name)

                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    class_prob_str = str(class_prob)
                    # remove the dot from the string
                    class_prob_str = class_prob_str.replace('.', '')
                    filename = f"frame_{frame_number}_{class_name}_{class_prob_str}_roi_{roi_count}.jpg"
                    save_path = os.path.join(directory, filename)
                    cv2.imwrite(save_path, cv2.cvtColor(frame_sign_rois[i], cv2.COLOR_RGB2BGR))

                    roi_count += 1

                    if class_prob > 0.9: predicted_classes.append(class_name)
                    else: predicted_classes.append('no_sign')
                    # if class_index != 1 and class_index != 2 and class_prob > 0.995:
                        # Zeichne die Bounding Box um das Schild
                    prediction = class_names[class_index] + " " + class_prob_str
                    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                    cv2.putText(frame, prediction, ((x_start - 10, y_start)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                predicted_classes.append('no_sign')


                # Write the frame number and predicted classes to the CSV file
            writer.writerow([frame_number] + predicted_classes)
            predicted_classes.clear()

            output_video.write(frame)

            if frame_number % 100 == 0:
                print(f"Processed {frame_number} frames")

    end_time = time.time()  # Stoppe die Zeitmessung
    elapsed_time = end_time - start_time  # Berechne die vergangene Zeit

    # Freigabe von Ressourcen
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

    print("Model: ", cnn_model_path)
    print(f"Time taken in total: {round(elapsed_time, 4)} seconds")
    print(f"Took an average of {round(haar_total_time/haar_counter_predictions, 4)} seconds for HAAR Model to classify an image - total time: {round(haar_total_time, 4)} - total predictions: {haar_counter_predictions}")
    print(f"Took an average of {round(cnn_total_time/cnn_counter_predictions, 4)} seconds for CNN Model to classify an image - total time: {round(cnn_total_time, 4)} - total predictions: {cnn_counter_predictions}")
    print(f"Other processing took {round(elapsed_time - haar_total_time - cnn_total_time, 4)} seconds.")


def classify_image(image_path, cnn_model_path, cascade_path):
    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    # Lade das Bild
    frame = cv2.imread(image_path)

    class_names = ['end_speed', 'no_sign', 'no_speed_sign', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

    save_roi_path = os.path.dirname(image_path) + "\\rois"
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wende das Haar Cascade-Modell auf das Bild an, um Schilder zu erkennen
    signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    start_time = time.time()  # Starte die Zeitmessung

    frame_sign_rois = np.zeros((len(signs), 128, 128, 3), dtype=np.uint8)
    coords = []
    # Durchlaufe erkannte Schilder
    for i, (x, y, w, h) in enumerate(signs):
        # Bereinige die Koordinaten der Bounding Box
        if (y - (h) < 0) or (y + (h) > 1080) or (x - (w) < 0) or x + (w) > 1920:
            continue

        # Schneide die Bounding Box aus dem Bild aus
        sign_roi = frame[y - h:y + h, x - w:x + w]

        # Reduziere die Größe um 10% (optional, falls benötigt)
        width_reduction = int(w * 0.4)
        height_reduction = int(h * 0.4)

        sign_roi_cropped = sign_roi[height_reduction:-height_reduction, width_reduction:-width_reduction]

        # Konvertiere das ROI in RGB und skaliere es
        frame_sign_rois[i] = cv2.resize(cv2.cvtColor(sign_roi_cropped, cv2.COLOR_BGR2RGB), (128, 128))

        coords.append((x, y, w, h))

        if frame_sign_rois.shape[0]:
                    predictions = cnn_model(frame_sign_rois)

        predicted_classes = []

        roi_count = 0
        # Schreibe das Frame in das Ausgabevideo
        for i, (x, y, w, h) in enumerate(coords):
            # Berechne die Koordinaten der Bounding Box nach der Reduzierung der Größe

            class_index = np.argmax(predictions[i])
            class_prob = np.round(np.max(predictions[i]),4)
            x_start = x - w + int(w * 0.1)
            y_start = y - h + int(h * 0.1)
            x_end = x + w - int(w * 0.1)
            y_end = y + h - int(h * 0.1)

            # save the bounding box to the folder with predicted class
            class_name = class_names[class_index]

            directory = os.path.join(save_roi_path, class_name)

            if not os.path.exists(directory):
                os.makedirs(directory)

            filename = f"{class_name}_{class_prob}_{roi_count}.jpg"
            save_path = os.path.join(directory, filename)
            cv2.imwrite(save_path, cv2.cvtColor(frame_sign_rois[i], cv2.COLOR_RGB2BGR))

            roi_count += 1

            if class_prob > 0.9: predicted_classes.append(class_name)
            # if class_index != 1 and class_index != 2 and class_prob > 0.995:
                # Zeichne die Bounding Box um das Schild
            prediction = class_names[class_index] + " " + str(class_prob)
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame, prediction, ((x_start - 10, y_start)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Zeige das Bild an
    cv2.imshow("Classified Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def classify_camera_stream(cnn_model_path):
    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade_path = "localization_models\LBP_7000_01_7\cascade.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    class_names = ['end_speed', 'no_sign', 'no_speed_sign', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

    # Öffne die Kamera
    cap = cv2.VideoCapture(0)  # 0 für die standardmäßige Kamera, kannst du auch andere Werte wie 1, 2 usw. verwenden

    while True:
        # Lese ein Frame aus der Kamera
        ret, frame = cap.read()
        if not ret:
            break

        # Wende das Haar Cascade-Modell auf das Frame an, um Schilder zu erkennen
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        signs = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

        # Durchlaufe erkannte Schilder
        for (x, y, w, h) in signs:
            # Schneide die Bounding Box aus dem Bild aus
            if (y - (h) < 0) or (y + (h) > 1080) or (x - (w) < 0) or x + (w) > 1920:
                continue

            sign_roi = frame[y - (h):y + (h), x - (w):x + (w)]
            width_reduction = int(w * 0.3)
            height_reduction = int(h * 0.3)
            sign_roi_cropped = sign_roi[height_reduction:-height_reduction, width_reduction:-width_reduction]

            # Konvertiere das ROI in RGB und skaliere es
            color_roi = cv2.cvtColor(sign_roi_cropped, cv2.COLOR_BGR2RGB)
            sign_roi_rescaled = cv2.resize(color_roi, (128, 128))

            # Klassifiziere das Schild mit dem CNN-Modell
            #predictions = cnn_model.predict(np.expand_dims(sign_roi_rescaled, axis=0))
            predictions = cnn_model(np.expand_dims(sign_roi_rescaled, axis=0), training=False)
            class_index = np.argmax(predictions)
            class_prob = np.round(np.max(predictions),4)
            class_name = class_names[class_index]

            if class_index != 1 and class_index != 2:
                if class_prob > 0.9:
                    prediction = class_name[class_index] + " " + str(class_prob)
                    # Zeichne die Bounding Box und das Label auf das Frame
                    cv2.rectangle(frame, (x - w, y - (h)), (x + w, y + (h)), (0, 255, 0), 2)
                    cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Zeige das Frame an
        cv2.imshow("Classified Camera Stream", frame)

        # Beende die Schleife, wenn 'q' gedrückt wird
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Freigabe von Ressourcen
    cap.release()
    cv2.destroyAllWindows()


def test_accuracy(parent_folder, cnn_model_path, cascade_path):
    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    class_name = ['end_speed', 'no_sign', 'no_speed_sign', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

    total_images = 0
    correct_predictions = 0
    incorrect_predictions = 0
    confusion_matrix = {}

    # Durchlaufen aller Unterordner im Hauptordner
    for folder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, folder_name, 'frames')

        # Überprüfen, ob der Unterordner 'frames' existiert
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):

            # Ground truth label aus dem Ordner-Namen extrahieren
            ground_truth_label = folder_name

            # Kopieren aller Bilder aus dem 'frames'-Ordner in den neuen Ordner
            for image_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, image_name)
                image = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
                signs = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

                # Durchlaufe erkannte Schilder
                for (x, y, w, h) in signs:
                    # Schneide die Bounding Box aus dem Bild aus
                    if (y - (h) < 0) or (y + (h) > 1080) or (x - w < 0) or x + w > 1920:
                        continue

                    sign_roi = image[y - (h):y + (h), x - w:x + w]
                    sign_roi_rescaled = cv2.resize(sign_roi, (128, 128))

                    # Klassifiziere das Schild mit dem CNN-Modell
                    #predictions = cnn_model.predict(np.expand_dims(sign_roi_rescaled, axis=0))
                    predictions = cnn_model(np.expand_dims(sign_roi_rescaled, axis=0), training=False)
                    class_index = np.argmax(predictions)
                    predicted_class = class_name[class_index]

                    total_images += 1
                    if predicted_class == ground_truth_label:
                        correct_predictions += 1
                    else:
                        incorrect_predictions += 1
                        # Update confusion matrix
                        if ground_truth_label not in confusion_matrix:
                            confusion_matrix[ground_truth_label] = {}
                        if predicted_class not in confusion_matrix[ground_truth_label]:
                            confusion_matrix[ground_truth_label][predicted_class] = 1
                        else:
                            confusion_matrix[ground_truth_label][predicted_class] += 1

    accuracy = correct_predictions / total_images
    print(f"Total Images: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Incorrect Predictions: {incorrect_predictions}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nConfusion Matrix:")
    for true_label, predictions in confusion_matrix.items():
        print(f"True Label: {true_label}")
        for predicted_label, count in predictions.items():
            print(f"  Predicted Label: {predicted_label}, Count: {count}")


if __name__ == "__main__":

    cnn_model_path = "Aaron\models\MobileNet.h5"
    classify_camera_stream(cnn_model_path)

#     #cascade_path = r'Dominik\cascade_12\cascade.xml'
#     cascade_path = r'localization_models\LBP_7000_01_7\cascade.xml'

#     cnn_model_path_deeper = r'Aaron\models\own_model_deeper.h5'
#     cnn_model_path_shallow = r'Aaron\models\own_model_shallow.h5'
#     cnn_model_path_mobileNet = r'Aaron\models\MobileNet.h5'
#     cnn_model_path_mobileNet60k = r'Aaron\models\MobileNet60k.h5'
#     cnn_model_path_mobileNet100k = r'Aaron\models\MobileNet100k.h5'
#     cnn_model_path_efficientNet60k = r'Aaron\models\EfficientNetB2_60k.h5'
#     cnn_model_path_efficientNet100k = r'Aaron\models\EfficientNetB2_100k.h5'

#     #classify_video_batch(video_path2, output_video_path4, cnn_model_path_mobileNet, cascade_path)

#     test_video_folder = r"test_videos"
#     result_video_folder = r"test_video_results_augmented_mobile"

#     # # Stelle sicher, dass das Ergebnisverzeichnis existiert
#     # if not os.path.exists(result_video_folder):
#     #     os.makedirs(result_video_folder)

#     # for video_file in os.listdir(test_video_folder):
#     #     # Überspringe, wenn es sich nicht um eine Videodatei handelt
#     #     if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
#     #         continue

# #         input_video_path = os.path.join(test_video_folder, video_file)
# #         output_video_path = os.path.join(result_video_folder, "processed_" + video_file)
#     video_numbers = [
#     "GX010093",
#     # "GX010098", "GX010103",
#     # "GX010094", "GX010099", "GX010105",
#     # "GX010095", "GX010100", "GX010106",
#     # "GX010096", "GX010101", "GX010107_d",
#     # "GX010097", "GX010102", "GX010108_d"
# ]
#     all_start_time = time.time()  # Starte die Zeitmessung

#     # Own Model shallow -> no augmentation
#     print("\n\n\nStarting to classify videos with Own Model shallow no augmentation")
#     start_time = time.time()  # Starte die Zeitmessung
#     for video_number in video_numbers:
#         input_video_path = rf"test_videos_with_labels\{video_number}.MP4"
#         output_video_path = rf"test_video_results_own_shallow\tests\{video_number}\classified_{video_number}.MP4"
#         print("Processing video: ", video_number)
#         classify_video_batch(input_video_path, output_video_path, cnn_model_path_shallow, cascade_path)
#         print("Finished processing video: ", video_number)
#     print("Classifying all videos took: ", time.time() - start_time, " seconds")

#     #  Own Model deeper -> no augmentation
#     print("\n\n\nStarting to classify videos with Own Model deeper no augmentation")
#     start_time = time.time()  # Starte die Zeitmessung
#     for video_number in video_numbers:
#         input_video_path = rf"test_videos_with_labels\{video_number}.MP4"
#         output_video_path = rf"test_video_results_own_deeper\tests\{video_number}\classified_{video_number}.MP4"
#         print("Processing video: ", video_number)
#         classify_video_batch(input_video_path, output_video_path, cnn_model_path_deeper, cascade_path)
#         print("Finished processing video: ", video_number)
#     print("Classifying all videos took: ", time.time() - start_time, " seconds")


#     # MobileNet -> no augmentation
#     print("\n\n\nStarting to classify videos with MobileNet no augmentation")
#     start_time = time.time()  # Starte die Zeitmessung
#     for video_number in video_numbers:
#         input_video_path = rf"test_videos_with_labels\{video_number}.MP4"
#         output_video_path = rf"test_video_results_MobileNet\tests\{video_number}\classified_{video_number}.MP4"
#         print("Processing video: ", video_number)
#         classify_video_batch(input_video_path, output_video_path, cnn_model_path_mobileNet, cascade_path)
#         print("Finished processing video: ", video_number)
#     print("Classifying all videos took: ", time.time() - start_time, " seconds")

#     # MobileNet60k -> augmentation with 60k images
#     print("\n\n\nStarting to classify videos with MobileNet60k")
#     start_time = time.time()  # Starte die Zeitmessung
#     for video_number in video_numbers:
#         input_video_path = rf"test_videos_with_labels\{video_number}.MP4"
#         output_video_path = rf"test_video_results_MobileNet60k\tests\{video_number}\classified_{video_number}.MP4"
#         print("Processing video: ", video_number)
#         classify_video_batch(input_video_path, output_video_path, cnn_model_path_mobileNet60k, cascade_path)
#         print("Finished processing video: ", video_number)
#     print("Classifying all videos took: ", time.time() - start_time, " seconds")


#     # MobileNet100k -> augmentation with 100k images
#     print("\n\n\nStarting to classify videos with MobileNet100k")
#     start_time = time.time()  # Starte die Zeitmessung
#     for video_number in video_numbers:
#         input_video_path = rf"test_videos_with_labels\{video_number}.MP4"
#         output_video_path = rf"test_video_results_MobileNet100k\tests\{video_number}\classified_{video_number}.MP4"
#         print("Processing video: ", video_number)
#         classify_video_batch(input_video_path, output_video_path, cnn_model_path_mobileNet100k, cascade_path)
#         print("Finished processing video: ", video_number)
#     print("Classifying all videos took: ", time.time() - start_time, " seconds")

#     # EfficientNetB2 -> augmentation with 60k images
#     print("\n\n\nStarting to classify videos with EfficientNetB2 60k")
#     start_time = time.time()  # Starte die Zeitmessung
#     for video_number in video_numbers:
#         input_video_path = rf"test_videos_with_labels\{video_number}.MP4"
#         output_video_path = rf"test_video_results_EfficientNet60k\tests\{video_number}\classified_{video_number}.MP4"
#         print("Processing video: ", video_number)
#         classify_video_batch(input_video_path, output_video_path, cnn_model_path_efficientNet60k, cascade_path)
#         print("Finished processing video: ", video_number)
#     print("Classifying all videos took: ", time.time() - start_time, " seconds")

#     # EfficientNetB2 -> augmentation with 100k images
#     print("\n\n\nStarting to classify videos with EfficientNetB2 100k")
#     start_time = time.time()  # Starte die Zeitmessung
#     for video_number in video_numbers:
#         input_video_path = rf"test_videos_with_labels\{video_number}.MP4"
#         output_video_path = rf"test_video_results_EfficientNet100k\tests\{video_number}\classified_{video_number}.MP4"
#         print("Processing video: ", video_number)
#         classify_video_batch(input_video_path, output_video_path, cnn_model_path_efficientNet100k, cascade_path)
#         print("Finished processing video: ", video_number)
#     print("Classifying all videos took: ", time.time() - start_time, " seconds")

#     print("Classifying all videos took: ", time.time() - all_start_time, " seconds")

