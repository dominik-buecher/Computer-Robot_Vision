import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os
import datetime


def classify_video(video_path, output_video_path, cnn_model_path, cascade_path):
    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    # Öffne das Video
    cap = cv2.VideoCapture(video_path)

    # Erstelle einen VideoWriter für das Ergebnisvideo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # oder 'XVID' je nach Codec
    output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    class_name = ['end_speed', 'no_sign', 'no_speed_sign', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']
    total_time = 0
    counter_predictions = 0
    start_time = time.time()  # Starte die Zeitmessung
    while True:
        # Lese ein Frame aus dem Video
        ret, frame = cap.read()
        if not ret:
            break

        # Wende das Haar Cascade-Modell auf das Frame an, um Schilder zu erkennen
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

        # Klassifiziere das Schild mit dem CNN-Modell
        # Durchlaufe erkannte Schilder
        for (x, y, w, h) in signs:
             # Schneide die Bounding Box aus dem Bild aus
            if (y - (h) < 0) or (y + (h) > 1080) or (x - (w) < 0) or x + (w) > 1920:
                continue

            # Schneide die Bounding Box aus dem Bild aus
            sign_roi = frame[y - h:y + h, x - w:x + w]

            # Berechne die Reduzierung der Größe für jede Dimension
            width_reduction = int(w * 0.1)
            height_reduction = int(h * 0.1)

            # Schneide das ROI, um es um 20% kleiner zu machen
            sign_roi_cropped = sign_roi[height_reduction:-height_reduction, width_reduction:-width_reduction]

            sign_roi_rescaled = cv2.resize(sign_roi_cropped, (128, 128))
            sign_roi_rescaled = cv2.cvtColor(sign_roi_rescaled, cv2.COLOR_BGR2RGB)

            predictions = cnn_model(np.expand_dims(sign_roi_rescaled, axis=0))
            
            class_index = np.argmax(predictions)
            prediction = class_name[class_index] + " " + str(np.max(predictions))
            prediction = round(prediction, 4)
            print(predictions)
            cv2.imshow("Classified Image", sign_roi_rescaled)
            cv2.waitKey(0)

            # Zeichne die Bounding Box und das Label auf das Bild
            # Berechne die neuen Koordinaten für die Bounding Box, die der verkleinerten ROI entsprechen
            cv2.rectangle(frame, (x - w + width_reduction, y - h + height_reduction), (x + w - width_reduction, y + h - height_reduction), (0, 0, 255), 3)
            cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

        # Schreibe das Frame in das Ausgabevideo
        output_video.write(frame)

    end_time = time.time()  # Stoppe die Zeitmessung
    elapsed_time = end_time - start_time  # Berechne die vergangene Zeit

    print(f"Time taken in totoal: {elapsed_time} seconds")
    print("Model: ", cnn_model_path)
    #print(f"Took an average of {total_time/counter_predictions} seconds for CNN Model to classify an image")
    # Freigabe von Ressourcen
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


def classify_video_batch(video_path, output_video_path, cnn_model_path, cascade_path):
        # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    # Öffne das Video
    cap = cv2.VideoCapture(video_path)

    # Erstelle einen VideoWriter für das Ergebnisvideo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # oder 'XVID' je nach Codec
    output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    class_names = ['end_speed', 'no_sign', 'no_speed_sign', '100', '120', '30', '40', '50', '70', '80']

    save_roi_path = os.path.dirname(output_video_path) + "\\rois"

    total_time = 0
    counter_predictions = 0
    roi_count = 0
    start_time = time.time()  # Starte die Zeitmessung
    while True:
        # Lese ein Frame aus dem Video
        ret, frame = cap.read()
        if not ret:
            break

        # Wende das Haar Cascade-Modell auf das Frame an, um Schilder zu erkennen
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

        frame_sign_rois = np.zeros((len(signs), 128, 128, 3), dtype=np.uint8)
        coords = []
        for i, (x, y, w, h) in enumerate(signs):
            # Bereinige die Koordinaten der Bounding Box
            if (y - (h) < 0) or (y + (h) > 1080) or (x - (w) < 0) or x + (w) > 1920:
                continue

            # Schneide die Bounding Box aus dem Bild aus
            sign_roi = frame[y - h:y + h, x - w:x + w]

            # Reduziere die Größe um 10% (optional, falls benötigt)
            width_reduction = int(w * 0.1)
            height_reduction = int(h * 0.1)

            sign_roi_cropped = sign_roi[height_reduction:-height_reduction, width_reduction:-width_reduction]

            # Konvertiere das ROI in RGB und skaliere es
            frame_sign_rois[i] = cv2.resize(cv2.cvtColor(sign_roi_cropped, cv2.COLOR_BGR2RGB), (128, 128))

            coords.append((x, y, w, h))

        if frame_sign_rois.shape[0]:
            predictions = cnn_model(frame_sign_rois)

        # Schreibe das Frame in das Ausgabevideo
        for i, (x, y, w, h) in enumerate(coords):
            # Berechne die Koordinaten der Bounding Box nach der Reduzierung der Größe

            class_index = np.argmax(predictions[i])
            class_prob = np.max(predictions[i])
            x_start = x - w + int(w * 0.1)
            y_start = y - h + int(h * 0.1)
            x_end = x + w - int(w * 0.1)
            y_end = y + h - int(h * 0.1)

            # save the bounding box to the folder with predicted class
            class_name = class_names[class_index]
            directory = os.path.join(save_roi_path, class_name)

            if not os.path.exists(directory):
                os.makedirs(directory)


            filename = f"{class_name}_{roi_count}_{class_prob}.jpg"
            save_path = os.path.join(directory, filename)
            cv2.imwrite(save_path, frame_sign_rois[i])
            roi_count += 1

            if class_index != 1 and class_index != 2 and class_prob > 0.995:
                # Zeichne die Bounding Box um das Schild
                prediction = class_names[class_index] + " " + str(class_prob)
                prediction = round(prediction, 4)
                cv2.rectangle(frame, (x - w + width_reduction, y - h + height_reduction), (x + w - width_reduction, y + h - height_reduction), (0, 0, 255), 3)
                cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

        output_video.write(frame)

    end_time = time.time()  # Stoppe die Zeitmessung
    elapsed_time = end_time - start_time  # Berechne die vergangene Zeit

    print(f"Time taken in totoal: {elapsed_time} seconds")
    print("Model: ", cnn_model_path)
    #print(f"Took an average of {total_time/counter_predictions} seconds for CNN Model to classify an image")
    # Freigabe von Ressourcen
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


def classify_image(image_path, output_image_path, cnn_model_path, cascade_path):
    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    # Lade das Bild
    frame = cv2.imread(image_path)

    class_name = ['end_speed', 'no_sign', 'no_speed_sign', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

    # Wende das Haar Cascade-Modell auf das Bild an, um Schilder zu erkennen
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    signs = cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    start_time = time.time()

    # Durchlaufe erkannte Schilder
    for (x, y, w, h) in signs:
        # Schneide die Bounding Box aus dem Bild aus
        sign_roi = frame[y - h:y + h, x - w:x + w]

        # Berechne die Reduzierung der Größe für jede Dimension
        width_reduction = int(w * 0.1)
        height_reduction = int(h * 0.1)

        # Schneide das ROI, um es um 20% kleiner zu machen
        sign_roi_cropped = sign_roi[height_reduction:-height_reduction, width_reduction:-width_reduction]
            # Zeige das Bild an
        cv2.imshow("cropped Image", sign_roi_cropped)

        sign_roi_rescaled = cv2.resize(sign_roi_cropped, (128, 128))
        sign_roi_rescaled = cv2.cvtColor(sign_roi_rescaled, cv2.COLOR_BGR2RGB)

        # Klassifiziere das Schild mit dem CNN-Modell
        start_time = time.time()  # Starte die Zeitmessung
        predictions = cnn_model.predict(np.expand_dims(sign_roi_rescaled, axis=0))
        end_time = time.time()  # Stoppe die Zeitmessung
        elapsed_time = end_time - start_time  # Berechne die vergangene Zeit

        print(f"Time taken for classification: {elapsed_time} seconds")

        class_index = np.argmax(predictions)
        prediction = class_name[class_index] + " " + str(np.max(predictions))
        prediction = round(prediction, 4)
        # Zeichne die Bounding Box und das Label auf das Bild
        # Berechne die neuen Koordinaten für die Bounding Box, die der verkleinerten ROI entsprechen
        cv2.rectangle(frame, (x - w + width_reduction, y - h + height_reduction), (x + w - width_reduction, y + h - height_reduction), (0, 0, 255), 3)
        cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

    # Zeige das Bild an
    cv2.imshow("Classified Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def classify_camera_stream(cnn_model_path, cascade_path):
    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    class_name = ['end_speed', 'no_sign', 'no_speed_sign', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

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
            sign_roi_rescaled = cv2.resize(sign_roi, (128, 128))

            # Klassifiziere das Schild mit dem CNN-Modell
            #predictions = cnn_model.predict(np.expand_dims(sign_roi_rescaled, axis=0))
            predictions = cnn_model(np.expand_dims(sign_roi_rescaled, axis=0), training=False)
            class_index = np.argmax(predictions)
            prediction = class_name[class_index]
            prediction = round(prediction, 4)
            if class_index != 1:
                # Zeichne die Bounding Box und das Label auf das Frame
                cv2.rectangle(frame, (x - w, y - (h)), (x + w, y + (h)), (0, 0, 255), 3)
                cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

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
    cascade_path = r'Dominik\cascade_12\cascade.xml'
    cnn_model_path_deeper = r'Aaron\models\own_model_deeper.h5'
    cnn_model_path_shallow = r'Aaron\models\own_model_shallow.h5'
    cnn_model_path_mobileNet = r'Aaron\models\MobileNetAugmented.h5'

    #classify_video_batch(video_path2, output_video_path4, cnn_model_path_mobileNet, cascade_path)

    test_video_folder = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_videos"
    result_video_folder = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_video_results_augmented_mobile"

    # # Stelle sicher, dass das Ergebnisverzeichnis existiert
    # if not os.path.exists(result_video_folder):
    #     os.makedirs(result_video_folder)

    # for video_file in os.listdir(test_video_folder):
    #     # Überspringe, wenn es sich nicht um eine Videodatei handelt
    #     if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
    #         continue

    #     input_video_path = os.path.join(test_video_folder, video_file)
    #     output_video_path = os.path.join(result_video_folder, "processed_" + video_file)
input_video_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_videos\no_audio_GX010093.MP4"
output_video_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\test_video_results_augmented_mobile\tests\GX010093\classified_GX010093.MP4"

classify_video_batch(input_video_path, output_video_path, cnn_model_path_mobileNet, cascade_path)