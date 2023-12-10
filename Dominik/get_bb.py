import cv2
import os

def extract_and_save_bounding_boxes(txt_file_path, input_image_folder, output_folder):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        values = line.strip().split(',')
        image_name = values[0]
        x, y, width, height = map(int, values[2:])
        
        image_path = os.path.join(input_image_folder, image_name)
        image = cv2.imread(image_path)
        
        if x-width < 0:
            xm = 0
        else:
            xm = x-width
        
        if x+width > 1920:
            xp = 1920
        else:
            xp = x + width

        if y-height < 0:
            ym = 0
        else:
            ym = y-height
        
        if y+height > 1080:
            yp = 1920
        else:
            yp = y+height

        # Schneide die Bounding-Box aus dem Bild aus
        bounding_box = image[ym:yp, xm:xp]
        print("image_name: ", image_name)
        # Speichere den ausgeschnittenen Bereich
        output_path = os.path.join(output_folder, f"{image_name[:-4]}_cropped_{x}_{y}_{width}_{height}.png")
        cv2.imwrite(output_path, bounding_box)

if __name__ == "__main__":
    txt_file_path = 'dataset/positive_samples/output.txt'
    input_image_folder = 'dataset/positive_samples'
    output_folder = 'dataset/boundingbox'

    extract_and_save_bounding_boxes(txt_file_path, input_image_folder, output_folder)
