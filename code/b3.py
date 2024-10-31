# b3.py
# extracting descriptors around each keypoint

import os
import sys
import numpy as np
from PIL import Image


def extract_descriptors(image_path, coords_path, output_dir):
    print(f"Processing image: {image_path}")
    im = Image.open(image_path).convert('L')
    im_gray = np.array(im, dtype=np.float32)
    height, width = im_gray.shape
    print("Image loaded and converted to grayscale.")

    coords = np.load(coords_path)
    print(f"Loaded {coords.shape[1]} corners.")

    descriptors = []
    valid_coords = []
    # ... half of 40x40 window size
    half_window_size = 20
    # ... get 8x8 descriptor
    downsample_factor = 5

    descriptor_images_dir = os.path.join(output_dir, 'descriptors_visualization')
    os.makedirs(descriptor_images_dir, exist_ok=True)

    for idx in range(coords.shape[1]):
        y, x = int(coords[0, idx]), int(coords[1, idx])
        if y - half_window_size < 0 or y + half_window_size > height or \
                x - half_window_size < 0 or x + half_window_size > width:
            continue

        patch = im_gray[y - half_window_size:y + half_window_size, x - half_window_size:x + half_window_size]

        descriptor_patch = patch[::downsample_factor, ::downsample_factor]

        descriptor = descriptor_patch.flatten()

        mean = np.mean(descriptor)
        std = np.std(descriptor)
        if std == 0:
            continue
        descriptor = (descriptor - mean) / std

        descriptors.append(descriptor)
        valid_coords.append([y, x])

        descriptor_visual = (descriptor_patch - descriptor_patch.min()) / (
                    descriptor_patch.max() - descriptor_patch.min())
        descriptor_visual = (descriptor_visual * 255).astype(np.uint8)
        descriptor_image = Image.fromarray(descriptor_visual)

        descriptor_image = descriptor_image.resize((80, 80), resample=Image.NEAREST)
        descriptor_image_filename = os.path.join(descriptor_images_dir, f'descriptor_{idx}.png')
        descriptor_image.save(descriptor_image_filename)
        print(f"Descriptor {idx} saved to {descriptor_image_filename}.")

    descriptors = np.array(descriptors)
    valid_coords = np.array(valid_coords).T

    print(f"Extracted {descriptors.shape[0]} descriptors.")

    descriptors_filename = os.path.join(output_dir, f'descriptors_{os.path.basename(image_path)}.npy')
    coords_filename = os.path.join(output_dir, f'valid_coords_{os.path.basename(image_path)}.npy')
    np.save(descriptors_filename, descriptors)
    np.save(coords_filename, valid_coords)
    print(f"Descriptors saved to {descriptors_filename}.")
    print(f"Valid coordinates saved to {coords_filename}.")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Error: Section argument is required.")
        sys.exit(1)

    section = sys.argv[1]
    # section = '12'

    images_dir = f'./images/{section}/'
    step2_dir = f'./part2_output/{section}/step2/'
    output_dir = f'./part2_output/{section}/step3/'
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {images_dir}.")
    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        coords_path = os.path.join(step2_dir, f'anms_corners_{image_name}.npy')
        extract_descriptors(image_path, coords_path, output_dir)
    print("Descriptor extraction completed.")
