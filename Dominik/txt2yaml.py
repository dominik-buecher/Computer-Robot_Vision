import yaml

def convert_to_yaml(input_file, output_file, image_path_prefix='C:/Users/Dominik/Documents/Studium/Master/Computer_vision/dataset/GTSDB/Test/Images_jpg/'):
    data = []

    with open(input_file, 'r') as input_file:
        for line in input_file:
            if line.startswith("Filename"):
                continue  # Überspringe die Header-Zeile
            fields = line.strip().split(';')
            filename, x1, y1, x2, y2, class_id, width, height = fields

            image_data = {
                'filename': f'image{filename.split(".")[0].zfill(3)}.jpg',
                'path': f'{image_path_prefix}image{filename.split(".")[0].zfill(3)}.jpg',
                'size': {
                    'width': int(width),
                    'height': int(height),
                    'depth': 3
                },
                'objects': [
                    {
                        'class': f'class_{class_id}',
                        'bounding_box': {
                            'xmin': int(x1),
                            'ymin': int(y1),
                            'xmax': int(x2),
                            'ymax': int(y2),
                        }
                    }
                ]
            }
            data.append(image_data)

    with open(output_file, 'w') as output_file:
        yaml.dump(data, output_file)

if __name__ == "__main__":
    input_file = r'C:\Users\Dominik\Documents\Studium\Master\Computer_vision\dataset\GTSDB\Test\Images_jpg\gt.txt'  # Ersetze dies durch den tatsächlichen Pfad zu deiner Textdatei
    output_file = 'Dominik/annotationen.yaml'  # Ersetze dies durch den gewünschten Ausgabepfad für deine YAML-Datei
    convert_to_yaml(input_file, output_file)

print(f'Daten wurden in die Datei {output_file} im YAML-Format geschrieben.')
