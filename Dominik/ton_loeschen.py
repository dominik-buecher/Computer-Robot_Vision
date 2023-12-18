import os
from moviepy.editor import VideoFileClip

def remove_audio(input_folder, output_folder):
    # Überprüfe, ob der Ausgabeordner existiert, andernfalls erstelle ihn
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Durchsuche den Eingabeordner nach Videodateien
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.MP4', '.mkv'))]

    # Iteriere über alle Videodateien im Eingabeordner
    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"no_audio_{video_file}")

        # Lade das Video
        video_clip = VideoFileClip(input_path)

        # Entferne den Ton
        video_clip = video_clip.set_audio(None)

        # Speichere das Video ohne Ton
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Schließe den Clip
        video_clip.close()

# Beispielaufruf
input_folder_path = 'videos'
output_folder_path = 'videos'

remove_audio(input_folder_path, output_folder_path)
