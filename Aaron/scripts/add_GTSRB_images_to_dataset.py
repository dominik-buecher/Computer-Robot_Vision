import os
import cv2
import random
import shutil

# Definition of the paths to the datasets
original_dataset_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\train"
gtsrb_dataset_path = r"C:\Users\aaron\Desktop\Programmierung\Master\Machine Vision\Computer-Robot_Vision_repo\datasets\sign_classification\GTSRB\Train\Images_jpg"

# Set maximum images per category
max_images = 2000

# Counting the number of images in each folder
category_image_count = {}
for category in os.listdir(original_dataset_path):
    path = os.path.join(original_dataset_path, category)
    if os.path.isdir(path) and "no_class" not in path:
        count = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        category_image_count[category] = count

# Print the number of images in each category
for category, count in category_image_count.items():
    print(f"Category '{category}' has {count} images.")

# Adding images from the GTSRB dataset to categories with less than max_images
for category in os.listdir(original_dataset_path):
    original_path = os.path.join(original_dataset_path, category)
    gtsrb_path = os.path.join(gtsrb_dataset_path, category)

    if os.path.isdir(original_path) and os.path.isdir(gtsrb_path):
        current_count = category_image_count[category]
        images_to_add = min(max_images - current_count, len(os.listdir(gtsrb_path)))

        print(f"Adding {images_to_add} images to category '{category}' to reach the maximum.")

        while current_count < max_images and images_to_add > 0:
            # Selecting a random image from the GTSRB dataset
            random_file = random.choice(os.listdir(gtsrb_path))
            src_file = os.path.join(gtsrb_path, random_file)
            dst_file = os.path.join(original_path, random_file)

            # Copying the image
            shutil.copy(src_file, dst_file)
            current_count += 1
            images_to_add -= 1
