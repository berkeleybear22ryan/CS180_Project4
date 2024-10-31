# b2.py
# detecting keypoints in each image


import os
import sys
import cv2
import numpy as np


def anms(coords, h, num_corners=500, c_robust=0.9):
    print("Performing ANMS...")
    ys, xs = coords[0], coords[1]
    N = ys.shape[0]
    radii = np.full(N, np.inf)

    corner_strengths = h[ys, xs]

    for i in range(N):
        xi, yi = xs[i], ys[i]
        strength_i = corner_strengths[i]
        stronger = np.where(corner_strengths > strength_i * c_robust)[0]
        if stronger.size > 0:
            xs_stronger = xs[stronger]
            ys_stronger = ys[stronger]
            distances = np.sqrt((xi - xs_stronger) ** 2 + (yi - ys_stronger) ** 2)
            radii[i] = distances.min()

    sorted_indices = np.argsort(-radii)
    selected_indices = sorted_indices[:num_corners]
    selected_coords = coords[:, selected_indices]
    print(f"ANMS selected {selected_coords.shape[1]} corners.")
    return selected_coords


def apply_anms(image_path, coords_path, harris_response_path, output_dir, num_corners=9000):
    print(f"Processing image: {image_path}")
    im_color = cv2.imread(image_path)
    if im_color is None:
        print(f"Error loading image {image_path}")
        return
    print("Image loaded successfully.")

    # make sure to load corner coordinates and harris response ...
    coords = np.load(coords_path)
    h = np.load(harris_response_path)
    print(f"Loaded {coords.shape[1]} corners.")

    coords = coords.astype(np.int32)

    selected_coords = anms(coords, h, num_corners=num_corners)

    selected_coords_filename = os.path.join(output_dir, f'anms_corners_{os.path.basename(image_path)}.npy')
    np.save(selected_coords_filename, selected_coords)
    print(f"Selected corners saved to {selected_coords_filename}.")

    print("Visualizing and saving image with selected corners...")
    im_with_corners = im_color.copy()
    for i in range(selected_coords.shape[1]):
        y, x = int(selected_coords[0, i]), int(selected_coords[1, i])
        cv2.circle(im_with_corners, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    output_image_path = os.path.join(output_dir, f'anms_corners_{os.path.basename(image_path)}')
    cv2.imwrite(output_image_path, im_with_corners)
    print(f"Image with selected corners saved to {output_image_path}.")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Error: Section argument is required.")
        sys.exit(1)

    section = sys.argv[1]
    # section = '12'

    images_dir = f'./images/{section}/'
    step1_dir = f'./part2_output/{section}/step1/'
    output_dir = f'./part2_output/{section}/step2/'
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {images_dir}.")
    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        coords_path = os.path.join(step1_dir, f'harris_corners_{image_name}.npy')
        harris_response_path = os.path.join(step1_dir, f'harris_response_{image_name}.npy')
        apply_anms(image_path, coords_path, harris_response_path, output_dir)
    print("ANMS processing completed.")
