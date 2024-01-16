import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def normalize_negative_one(img):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return 2*normalized_input - 1

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

# Get the path of the folder containing the images
folder_path = current_dir

# Initialize lists to store the original and scaled images
original_images = []
scaled_images = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct the full path of the image file
        image_path = os.path.join(folder_path, filename)

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Append the original image to the list
        original_images.append(image)

        # Scale the pixel values to the range [0, 1]
        scaled_image = normalize_negative_one(image)

        # Append the scaled image to the list
        scaled_images.append(scaled_image)

# Create a figure with two rows and the number of images as columns
fig, axes = plt.subplots(2, len(original_images))

# Plot the original images in the first row
for i, image in enumerate(original_images):
    axes[0, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, i].axis('off')

# Plot the scaled images in the second row
for i, image in enumerate(scaled_images):
    axes[1, i].imshow(image)
    axes[1, i].axis('off')

# Show the plot
plt.show()
