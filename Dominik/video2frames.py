import cv2
import os

def video_to_frames(video_path, output_path):
    # Öffne das Video
    video_capture = cv2.VideoCapture(video_path)
    
    # Überprüfe, ob das Video erfolgreich geöffnet wurde
    if not video_capture.isOpened():
        print("Fehler beim Öffnen des Videos.")
        return
    
    # Erstelle den Ausgabeordner, wenn er nicht existiert
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Initialisiere Variablen
    frame_count = 0
    
    # Schleife durch alle Frames im Video
    while True:
        # Lies den nächsten Frame
        ret, frame = video_capture.read()
        
        # Überprüfe, ob das Ende des Videos erreicht wurde
        if not ret:
            break
        
        # Speichere den Frame als .jpg-Bild ab
        frame_filename = f"frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_path, frame_filename)
        cv2.imwrite(frame_path, frame)
        
        # Inkrementiere den Frame-Zähler
        frame_count += 1

    # Gib eine Erfolgsmeldung aus
    print(f"{frame_count} Frames wurden erfolgreich extrahiert.")
    
    # Schließe das Video
    video_capture.release()

# Beispielaufruf des Skripts
video_path = r"videos_24_11_2023\video_speed_combined.mp4"
output_path = r"videos_24_11_2023\combined"
video_to_frames(video_path, output_path)
