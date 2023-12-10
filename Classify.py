import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os

def classify_video(video_path, output_video_path, cnn_model_path, cascade_path):
    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    # Öffne das Video
    cap = cv2.VideoCapture(video_path)

    # Erstelle einen VideoWriter für das Ergebnisvideo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # oder 'XVID' je nach Codec
    output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    class_name = ['end_speed', 'no_class', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

    while True:
        # Lese ein Frame aus dem Video
        ret, frame = cap.read()
        if not ret:
            break

        # Wende das Haar Cascade-Modell auf das Frame an, um Schilder zu erkennen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        signs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Durchlaufe erkannte Schilder
        for (x, y, w, h) in signs:

            if (y - (h // 2) < 0) or (y + (h // 2) > 1080) or (x - (w // 2) < 0) or x + (w // 2) > 1920:
                continue

            sign_roi = frame[y - (h // 2):y + (h // 2), x - (w // 2):x + (w // 2)]

            sign_roi_rescaled = cv2.resize(sign_roi, (128, 128))

            predictions = cnn_model.predict(np.expand_dims(sign_roi_rescaled, axis=0))
            class_index = np.argmax(predictions)
            prediction = class_name[class_index]

            if class_index != 1:
                # Zeichne die Bounding Box und das Label auf das Frame
                cv2.rectangle(frame, (x- (w//2), y - (h//2)), (x + (w//2), y + (h//2)), (0, 255, 0), 2)
                cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Schreibe das Frame in das Ausgabevideo
        output_video.write(frame)

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

    class_name = ['end_speed', 'no_class', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

    # Wende das Haar Cascade-Modell auf das Bild an, um Schilder zu erkennen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    start_time = time.time()  # Starte die Zeitmessung

    # Durchlaufe erkannte Schilder
    for (x, y, w, h) in signs:
        if (y - h < 0) or (y + h > 1080) or (x - w < 0) or x + w > 1920:
            continue

        # Schneide die Bounding Box aus dem Bild aus
        sign_roi = frame[y - (h // 2):y + (h // 2), x - (w // 2):x + (w // 2)]
        sign_roi_rescaled = cv2.resize(sign_roi, (128, 128))

        # Klassifiziere das Schild mit dem CNN-Modell
        predictions = cnn_model.predict(np.expand_dims(sign_roi_rescaled, axis=0))
        print("predictions: ", predictions)
        class_index = np.argmax(predictions)
        prediction = class_name[class_index]

        if class_index != 1:
            # Zeichne die Bounding Box und das Label auf das Bild
            cv2.rectangle(frame, (x - w, y - h), (x + w, y +h), (0, 255, 0), 2)
            cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    end_time = time.time()  # Stoppe die Zeitmessung
    elapsed_time = end_time - start_time  # Berechne die vergangene Zeit

    print(f"Time taken for classification: {elapsed_time} seconds")

    cv2.imwrite(output_image_path, frame)
    # Zeige das Bild an
    cv2.imshow("Classified Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def classify_camera_stream(cnn_model_path, cascade_path):
    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    class_name = ['end_speed', 'no_class', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

    # Öffne die Kamera
    cap = cv2.VideoCapture(0)  # 0 für die standardmäßige Kamera, kannst du auch andere Werte wie 1, 2 usw. verwenden

    while True:
        # Lese ein Frame aus der Kamera
        ret, frame = cap.read()
        if not ret:
            break

        # Wende das Haar Cascade-Modell auf das Frame an, um Schilder zu erkennen
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        signs = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Durchlaufe erkannte Schilder
        for (x, y, w, h) in signs:
            # Schneide die Bounding Box aus dem Bild aus
            if (y - (h // 2) < 0) or (y + (h // 2) > 1080) or (x - (w // 2) < 0) or x + (w // 2) > 1920:
                continue

            sign_roi = frame[y - (h // 2):y + (h // 2), x - (w // 2):x + (w // 2)]
            sign_roi_rescaled = cv2.resize(sign_roi, (128, 128))

            # Klassifiziere das Schild mit dem CNN-Modell
            predictions = cnn_model.predict(np.expand_dims(sign_roi_rescaled, axis=0))
            class_index = np.argmax(predictions)
            prediction = class_name[class_index]

            if class_index != 1:
                # Zeichne die Bounding Box und das Label auf das Frame
                cv2.rectangle(frame, (x - (w // 2), y - (h // 2)), (x + (w // 2), y + (h // 2)), (0, 255, 0), 2)
                cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Zeige das Frame an
        cv2.imshow("Classified Camera Stream", frame)

        # Beende die Schleife, wenn 'q' gedrückt wird
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Freigabe von Ressourcen
    cap.release()
    cv2.destroyAllWindows()


def test_accuracy(parent_folder):
    # Lade das Haar Cascade-Modell für die Schilderlokalisierung
    cascade_path = 'path/to/your/cascade.xml'  # Replace with the actual path to your cascade.xml file
    cascade = cv2.CascadeClassifier(cascade_path)

    # Lade das CNN-Modell für die Schilderklassifikation
    cnn_model_path = 'path/to/your/cnn_model.h5'  # Replace with the actual path to your cnn_model.h5 file
    cnn_model = tf.keras.models.load_model(cnn_model_path)

    class_name = ['end_speed', 'no_class', 'speed_100', 'speed_120', 'speed_30', 'speed_40', 'speed_50', 'speed_70', 'speed_80']

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
                signs = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Durchlaufe erkannte Schilder
                for (x, y, w, h) in signs:
                    # Schneide die Bounding Box aus dem Bild aus
                    if (y - (h // 2) < 0) or (y + (h // 2) > 1080) or (x - (w // 2) < 0) or x + (w // 2) > 1920:
                        continue

                    sign_roi = image[y - (h // 2):y + (h // 2), x - (w // 2):x + (w // 2)]
                    sign_roi_rescaled = cv2.resize(sign_roi, (128, 128))

                    # Klassifiziere das Schild mit dem CNN-Modell
                    predictions = cnn_model.predict(np.expand_dims(sign_roi_rescaled, axis=0))
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

    # video_path = r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\david\Aufnahme_01.MP4'
    # output_video_path = r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\david\Aufnahme_01_new.MP4'

    cascade_path = r'Dominik\cascade_12\cascade.xml'
    #cnn_model_path = r'Aaron\models\MobileNet.h5'
    #cnn_model_path = r'Aaron\models\own_model_deeper.h5'
    cnn_model_path = r'Aaron\models\own_model_deeper.h5'

    #classify_video(video_path, output_video_path, cnn_model_path, cascade_path)

    image_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\videos_24_11_2023\speed_70\frames\frame_0095.jpg"
    output_image_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\Aaron\results\result1.jpg"

    classify_image(image_path, output_image_path, cnn_model_path, cascade_path)