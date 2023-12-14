import cv2
import os

def video_to_frames(video_path, output_path):
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print("Fehler beim Öffnen des Videos.")
        return
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_filename = f"frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_path, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    print(f"{frame_count} Frames wurden erfolgreich extrahiert.")
    video_capture.release()


video_path = r"videos_24_11_2023\video_speed_combined.mp4"
output_path = r"videos_24_11_2023\combined"
video_to_frames(video_path, output_path)
