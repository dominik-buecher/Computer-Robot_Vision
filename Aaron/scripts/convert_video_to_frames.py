import cv2
import os

# Pfad zum übergeordneten Verzeichnis mit den Unterordnern für die MP4-Dateien
parent_folder = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\videos_24_11_2023"

# Liste der Unterordner im übergeordneten Verzeichnis
subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

# Schleife über die Unterordner
for subfolder in subfolders:
    # Pfad zum Ordner mit den MP4-Dateien
    input_folder = os.path.join(parent_folder, subfolder)

    # Erstelle den Unterordner für die extrahierten Frames
    output_folder = os.path.join(input_folder, "frames")
    os.makedirs(output_folder, exist_ok=True)

    # Liste der Dateien im Ordner
    file_list = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    # Funktion zum Extrahieren der Frames aus einer MP4-Datei
    def extract_frames(video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_filename = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            if frame_count % 100 == 0:
                print(f"Finished {frame_count} images of {video_path}")

        cap.release()

    # Extrahiere Frames aus allen MP4-Dateien im aktuellen Unterordner
    for file_name in file_list:
        video_path = os.path.join(input_folder, file_name)
        extract_frames(video_path, output_folder)

    print(f"Fertig mit '{subfolder}'. Frames wurden aus {len(file_list)} Videos extrahiert und im Ordner 'frames' abgelegt.")
