# dataset/positive_samples/00003.jpg 1 753 454 23 23


# def combine_lines(input_file, output_file):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         lines = infile.readlines()
#         i = 0
#         while i < len(lines) - 1:
#             current_line = lines[i].strip().split(',')
#             next_line = lines[i + 1].strip().split(',')

#             if current_line[0] == next_line[0]:
#                 combined_line = [current_line[0], str(int(current_line[1]) + 1)] + [current_line[2]] + current_line[3:] + next_line[2:]
#                 lines[i] = ' '.join(combined_line) + '\n'
#                 lines.pop(i + 1)
#             else:
#                 i += 1

#         # Ersetze alle Kommas durch Leerzeichen in den verbleibenden Zeilen
#         for j in range(len(lines)):
#             lines[j] = lines[j].replace(',', ' ')

#         outfile.writelines(lines)
# # Beispielaufruf
# combine_lines(r'dataset\positive_samples\output.txt', r'dataset\positive_samples\output2.txt')


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


# def add_prefix_to_file(input_file, output_file, prefix='dataset/positive_samples/'):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         for line in infile:
#             parts = line.strip().split()
#             if parts:
#                 parts[0] = f"{prefix}{parts[0]}"
#                 outfile.write(' '.join(parts) + '\n')

# # Beispielaufruf
# add_prefix_to_file(r'dataset\positive_samples\pos_own.txt', r'dataset\positive_samples\pos.txt')


# def remove_prefix_from_file(input_file, output_file, prefix='dataset/positive_samples/'):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         for line in infile:
#             parts = line.strip().split()
#             if parts and parts[0].startswith(prefix):
#                 parts[0] = parts[0][len(prefix):]
#                 outfile.write(' '.join(parts) + '\n')

# # Beispielaufruf
# remove_prefix_from_file(r'dataset\positive_samples\pos.txt', r'dataset\positive_samples\pos_new.txt')



# def adjust_spacing(input_file, output_file, spaces_between_segments=2):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         for line in infile:
#             parts = line.strip().split()
#             if parts:
#                 adjusted_line = parts[0] + ' ' * spaces_between_segments + ' '.join(parts[1:]) + {' ' * spaces_between_segments} + {' '.join(parts[2:])} + '\n'
#                 outfile.write(adjusted_line)





# # Beispielaufruf
# adjust_spacing(r'dataset\positive_samples\output2.txt', 'dataset\positive_samples\pos_new3.txt')




# def adjust_coordinates(input_file, output_file):
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         for line in infile:
#             parts = line.strip().split(',')
#             if len(parts) == 6:
#                 x_coordinate = int(parts[2])
#                 width = int(parts[4])

#                 # Überprüfe, ob die Koordinaten größer als 1919 sind
#                 if x_coordinate > 1919 or (x_coordinate + width) > 1919:
#                     # Passe die Koordinaten an, wenn nötig
#                     if x_coordinate > 1919:
#                         x_coordinate = 1919 - width
#                     if (x_coordinate + width) > 1919:
#                         width = 1919 - x_coordinate

#                 # Überprüfe, ob die Koordinaten kleiner als 1 sind
#                 if x_coordinate < 1 or (x_coordinate + width) < 1:
#                     # Passe die Koordinaten an, wenn nötig
#                     if x_coordinate < 1:
#                         x_coordinate = 1
#                     if (x_coordinate + width) < 1:
#                         width = 1 - x_coordinate

#                 # Erstelle die aktualisierte Zeile
#                 updated_line = f"{parts[0]},{parts[1]},{x_coordinate},{parts[3]},{width},{parts[5]}\n"
#                 outfile.write(updated_line)
#             else:
#                 # Schreibe Zeilen ohne das erwartete Format unverändert
#                 outfile.write(line)


# # Beispielaufruf
# adjust_coordinates(r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\own_dataset\combined\combined.txt', 'dataset/positive_samples/output.txt')

