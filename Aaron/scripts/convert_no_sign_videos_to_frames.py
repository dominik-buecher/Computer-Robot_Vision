import cv2
import os
import random

video_folder = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\videos_24_11_2023\no_class"
frame_folders = [] # Speichert Pfade zu den Ordnern mit Frames

# Schritt 1: Videos durchlaufen und Ordner erstellen
for video_file in os.listdir(video_folder):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_folder, video_file)
        frame_folder = os.path.join(video_folder, video_file.split('.')[0])
        frame_folders.append(frame_folder)

        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)

        # Schritt 2: Frames extrahieren
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(frame_folder, f'frame{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()

# Schritt 3: Frames aus allen Ordnern sammeln
min_frames = min(len(os.listdir(folder)) for folder in frame_folders)
frames_from_all_folder = os.path.join(video_folder, 'frames_from_all')

if not os.path.exists(frames_from_all_folder):
    os.makedirs(frames_from_all_folder)

for folder in frame_folders:
    selected_frames = random.sample(os.listdir(folder), min_frames)
    for frame_file in selected_frames:
        source_path = os.path.join(folder, frame_file)
        destination_path = os.path.join(frames_from_all_folder, f"{os.path.basename(folder)}_{frame_file}")
        os.rename(source_path, destination_path)

print(f"Vorgang abgeschlossen. {min_frames} Frames wurden aus jedem Ordner in '{frames_from_all_folder}' verschoben.")
