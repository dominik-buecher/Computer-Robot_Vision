import os
import cv2

folder_path = os.path.dirname(__file__)
video_files = [file for file in os.listdir(folder_path) if file.endswith(".MP4")]

total_frames = 0

for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    video = cv2.VideoCapture(video_path)
    total_frames += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

print("Total frames in all videos:", total_frames)
