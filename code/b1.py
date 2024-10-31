# b1.py
# reading and preparing images for processing


import os
import cv2
import numpy as np
from harris import get_harris_corners
import os
import sys


def detect_harris_corners(image_path, output_dir, edge_discard=20):
    print(f"Processing image: {image_path}")
    im_color = cv2.imread(image_path)
    if im_color is None:
        print(f"Error loading image {image_path}")
        return
    print("Image loaded successfully.")
    im_gray = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
    im_gray = np.float32(im_gray)
    print("Image converted to grayscale.")

    print("Starting Harris corner detection...")
    print(f"im_gray.shape: {im_gray.shape}")
    h, coords = get_harris_corners(im_gray, edge_discard=edge_discard, min_distance=1, num_peaks=10_000)
    print("Harris corner detection completed.")

    coords_filename = os.path.join(output_dir, f'harris_corners_{os.path.basename(image_path)}.npy')
    np.save(coords_filename, coords)
    print(f"Corner coordinates saved to {coords_filename}.")

    h_filename = os.path.join(output_dir, f'harris_response_{os.path.basename(image_path)}.npy')
    np.save(h_filename, h)
    print(f"Harris response saved to {h_filename}.")

    print("Visualizing and saving image with corners...")
    im_with_corners = im_color.copy()
    for i in range(coords.shape[1]):
        y, x = int(coords[0, i]), int(coords[1, i])
        cv2.circle(im_with_corners, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
    output_image_path = os.path.join(output_dir, f'harris_corners_{os.path.basename(image_path)}')
    cv2.imwrite(output_image_path, im_with_corners)
    print(f"Image with corners saved to {output_image_path}.")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Error: Section argument is required.")
        sys.exit(1)

    section = sys.argv[1]
    # section = '12'

    images_dir = f'./images/{section}/'
    output_dir = f'./part2_output/{section}/step1/'
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {images_dir}.")
    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        detect_harris_corners(image_path, output_dir)
    print("Processing completed.")
