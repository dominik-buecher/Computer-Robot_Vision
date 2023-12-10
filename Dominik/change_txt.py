# dataset/positive_samples/00003.jpg 1 753 454 23 23
import os
import shutil

# def combine_lines(input_file, output_file):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         lines = infile.readlines()
#         i = 0
#         while i < len(lines) - 1:
#             current_line = lines[i].strip().split(';')
#             next_line = lines[i + 1].strip().split(';')

#             if current_line[0] == next_line[0]:
#                 combined_line = [current_line[0], str(int(current_line[1]) + 1)] + [current_line[2]] + current_line[3:] + next_line[1:]
#                 lines[i] = ' '.join(combined_line) + '\n'
#                 lines.pop(i + 1)
#             else:
#                 i += 1

#         # Ersetze alle Kommas durch Leerzeichen in den verbleibenden Zeilen
#         for j in range(len(lines)):
#             lines[j] = lines[j].replace(';', ' ')

#         outfile.writelines(lines)
# # Beispielaufruf
# combine_lines(r'Dominik\gt copy2.txt', r'Dominik\pos_new.txt')


# import os

# def rename_images(folder_path, prefix='no_sign_'):
#     counter = 0

#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#             counter += 1
#             new_name = f"{prefix}{counter}"
#             file_path = os.path.join(folder_path, filename)
#             new_file_path = os.path.join(folder_path, f"{new_name}{os.path.splitext(filename)[1]}")
#             os.rename(file_path, new_file_path)

# # Beispielaufruf
# rename_images(r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\own_dataset\no_class\frames')

# import os

# def create_image_list_txt(folder_path, output_file, prefix='dataset/negative_samples/'):
#     with open(output_file, 'w') as outfile:
#         for filename in os.listdir(folder_path):
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#                 image_path = os.path.join(prefix, filename)
#                 outfile.write(image_path + '\n')

# # Beispielaufruf
# create_image_list_txt(r'dataset\negative_samples', r'dataset\negative_samples\neg.txt')



# frame_0003.jpg,1,611,744,25,24
# frame_0003.jpg,2,611,744,25,24frame_0004.jpg,1,604,743,25,24
# frame_0004.jpg,2,604,743,25,24frame_0005.jpg,1,599,743,26,25

# frame_0003.jpg,2,611,744,25,24,611,744,25,24
# frame_0004.jpg,2,604,743,25,24,604,743,25,24+


def add_prefix_to_file(input_file, output_file, prefix='dataset/negative_samples/'):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if parts:
                parts[0] = f"{prefix}{parts[0]}"
                outfile.write(' '.join(parts) + '\n')

# Beispielaufruf
add_prefix_to_file(r'dataset\negative_samples\neg.txt', r'dataset\negative_samples\neg2.txt')


# def remove_prefix_from_file(input_file, output_file, prefix='dataset/positive_samples/'):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         for line in infile:
#             parts = line.strip().split()
#             if parts and parts[0].startswith(prefix):
#                 parts[0] = parts[0][len(prefix):]
#                 outfile.write(' '.join(parts) + '\n')

# # Beispielaufruf
# remove_prefix_from_file(r'Dominik\pos.txt', r'Dominik\pos_new.txt')



# def adjust_spacing(input_file, output_file, spaces_between_segments=2):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         for line in infile:
#             parts = line.strip().split()
#             if parts:
#                 adjusted_line = parts[0] + ' ' * spaces_between_segments + ' '.join(parts[1:]) + {' ' * spaces_between_segments} + {' '.join(parts[2:])} + '\n'
#                 outfile.write(adjusted_line)





# # Beispielaufruf
# adjust_spacing(r'dataset\positive_samples\output2.txt', 'dataset\positive_samples\pos_new3.txt')




def adjust_coordinates(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(' ')
            if len(parts) == 6:
                x_coordinate = int(parts[2])
                width = int(parts[4])

                # Überprüfe, ob die Koordinaten größer als 1919 sind
                if x_coordinate > 1919 or (x_coordinate + width) > 1919:
                    # Passe die Koordinaten an, wenn nötig
                    if x_coordinate > 1919:
                        x_coordinate = 1919 - width
                    if (x_coordinate + width) > 1919:
                        width = 1919 - x_coordinate

                # Überprüfe, ob die Koordinaten kleiner als 1 sind
                if x_coordinate < 1 or (x_coordinate + width) < 1:
                    # Passe die Koordinaten an, wenn nötig
                    if x_coordinate < 1:
                        x_coordinate = 1
                    if (x_coordinate + width) < 1:
                        width = 1 - x_coordinate

                # Erstelle die aktualisierte Zeile
                updated_line = f"{parts[0]},{parts[1]},{x_coordinate},{parts[3]},{width},{parts[5]}\n"
                outfile.write(updated_line)
            else:
                # Schreibe Zeilen ohne das erwartete Format unverändert
                outfile.write(line)


# Beispielaufruf
#adjust_coordinates(r'Dominik\annotation_files\train.txt', r'Dominik\annotation_files\train2.txt')









def change_size(x_coordinate, y_coordinate, width, height):
    # Überprüfe, ob die Koordinaten größer als 1920 sind
    if x_coordinate > 1920 or (x_coordinate + width) > 1920:
        # Passe die Koordinaten an, wenn nötig
        if x_coordinate > 1920:
            x_coordinate = 1920 - width
        if (x_coordinate + width) > 1920:
            width = 1920 - x_coordinate

    # Überprüfe, ob die Koordinaten kleiner als 1 sind
    if x_coordinate < 1 or (x_coordinate + width) < 1:
        # Passe die Koordinaten an, wenn nötig
        if x_coordinate < 1:
            x_coordinate = 1
        if (x_coordinate + width) < 1:
            width = 1 - x_coordinate

            # Überprüfe, ob die Koordinaten größer als 1920 sind
    if y_coordinate > 1080 or (y_coordinate + height) > 1080:
        # Passe die Koordinaten an, wenn nötig
        if y_coordinate > 1080:
            y_coordinate = 1080 - height
        if (y_coordinate + height) > 1080:
            height = 1080 - y_coordinate

    # Überprüfe, ob die Koordinaten kleiner als 1 sind
    if y_coordinate < 1 or (y_coordinate + height) < 1:
        # Passe die Koordinaten an, wenn nötig
        if y_coordinate < 1:
            y_coordinate = 1
        if (y_coordinate + height) < 1:
            height = 1 - y_coordinate
    return x_coordinate, y_coordinate, width, height

def adjust_coordinates2(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    adjusted_lines = []

    for line in lines:
        elements = line.split()
        image_name = elements[0]
        class_label = elements[1]
        x_coordinate = int(elements[2])
        y_coordinate = int(elements[3])
        width = int(elements[4])
        height = int(elements[5])

        if len(elements) > 7:
            x_coordinate2 = int(elements[6])
            y_coordinate2 = int(elements[7])
            width2 = int(elements[8])
            height2 = int(elements[9])

            x_coordinate2, y_coordinate2, width2, height2 = change_size(x_coordinate2, y_coordinate2, width2, height2)
        
        x_coordinate, y_coordinate, width, height = change_size(x_coordinate, y_coordinate, width, height)
        if len(elements) > 7:
            adjusted_lines.append(f"{image_name} {class_label} {x_coordinate} {y_coordinate} {width} {height} {x_coordinate2} {y_coordinate2} {width2} {height2}\n")
        else:
            adjusted_lines.append(f"{image_name} {class_label} {x_coordinate} {y_coordinate} {width} {height}\n")

    with open(output_file, 'w') as output_file:
        output_file.writelines(adjusted_lines)

# Beispielaufruf
#adjust_coordinates2(r'dataset\positive_samples\train\train.txt', r'dataset\positive_samples\train\train2.txt')







# dateipfad = r'Dominik\pos_new.txt'
# bearbeitete_zeilen = []

# with open(dateipfad, 'r') as datei:
#     for zeile in datei:
#         teile = zeile.strip().split()
#         bildname = teile[0]
#         zahlen = teile[1:]
        
#         # Überprüfe, ob es vier oder mehr Zahlen gibt
#         kennzeichnung = "1" if len(zahlen) == 4 else "2"
        
#         neue_zeile = f"{bildname} {kennzeichnung} {' '.join(zahlen)}"
#         bearbeitete_zeilen.append(neue_zeile)

# with open(dateipfad, 'w') as datei:
#     for bearbeitete_zeile in bearbeitete_zeilen:
#         datei.write(f"{bearbeitete_zeile}\n")

# print("Bearbeitung abgeschlossen.")




def copy_images(input_file, source_folder, destination_folder):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        elements = line.split()
        image_name = elements[0]

        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)

        if os.path.isfile(source_path):
            shutil.copy(source_path, destination_path)
            print(f"Kopiere {image_name} nach {destination_folder}")
        else:
            print(f"{image_name} nicht gefunden im Quellordner {source_folder}")

# Beispielaufruf
input_txt = r'Dominik\annotation_files\test.txt'  # Passe dies entsprechend an
source_folder = r'dataset\positive_samples\combined'       # Passe dies entsprechend an
destination_folder = r'dataset\positive_samples\test'    # Passe dies entsprechend an

#copy_images(input_txt, source_folder, destination_folder)




def extract_image_names(folder_path, output_file):
    with open(output_file, 'w') as file:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file.write(f"{filename}\n")

# Beispielaufruf
folder_path = r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\own_dataset\no_class\frames'
output_file = r'Dominik\annotation_files\neg.txt'
#extract_image_names(folder_path, output_file)
