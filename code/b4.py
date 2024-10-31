# b4.py
# matching descriptors between images

import os
import sys
import numpy as np
from PIL import Image, ImageDraw


def hsv_to_rgb(h, s, v):
    h = h * 6
    i = int(h) % 6
    f = h - int(h)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    # i == 5
    else:
        r, g, b = v, p, q

    return r, g, b


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def match_descriptors(descriptors1, descriptors2, ratio_threshold=0.8):
    matches_12 = []
    matches_21 = []

    descriptors1 = descriptors1.astype(np.float32)
    descriptors2 = descriptors2.astype(np.float32)

    for i, desc1 in enumerate(descriptors1):
        diff = descriptors2 - desc1
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        if len(distances) < 2:
            continue
        sorted_indices = np.argsort(distances)
        nearest = sorted_indices[0]
        second_nearest = sorted_indices[1]
        if distances[nearest] / distances[second_nearest] < ratio_threshold:
            matches_12.append((i, nearest))

    for j, desc2 in enumerate(descriptors2):
        diff = descriptors1 - desc2
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        if len(distances) < 2:
            continue
        sorted_indices = np.argsort(distances)
        nearest = sorted_indices[0]
        second_nearest = sorted_indices[1]
        if distances[nearest] / distances[second_nearest] < ratio_threshold:
            matches_21.append((nearest, j))

    matches_12_set = set(matches_12)
    mutual_matches = [m for m in matches_21 if m in matches_12_set]

    return mutual_matches


def combine_images(image_paths):
    images = [Image.open(p).convert('RGB') for p in image_paths]
    widths, heights = zip(*(im.size for im in images))
    total_width = sum(widths)
    max_height = max(heights)
    combined_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    image_offsets = []
    for im in images:
        combined_image.paste(im, (x_offset, 0))
        image_offsets.append(x_offset)
        x_offset += im.size[0]

    return combined_image, image_offsets


def adjust_coordinates(coords_list, image_offsets):
    adjusted_coords_list = []
    for coords, offset in zip(coords_list, image_offsets):
        adjusted_coords = coords.copy()
        adjusted_coords[1, :] += offset
        adjusted_coords_list.append(adjusted_coords)
    return adjusted_coords_list


def match_all_images(image_files, descriptors_list, coords_list, output_dir):
    num_images = len(image_files)
    all_matches = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            descriptors1 = descriptors_list[i]
            descriptors2 = descriptors_list[j]
            matches = match_descriptors(descriptors1, descriptors2, ratio_threshold=0.9)
            all_matches.append((i, j, matches))
            print(f"Matched {image_files[i]} with {image_files[j]}: {len(matches)} matches.")

            matches_filename = os.path.join(output_dir, f'matches_{image_files[i]}_{image_files[j]}.npy')
            np.save(matches_filename, matches)
            print(f"Matches saved to {matches_filename}.")

    return all_matches


def visualize_all_matches(image_paths, coords_list, all_matches, output_path):
    print("Creating combined image and visualizing all matches...")
    combined_image, image_offsets = combine_images(image_paths)
    adjusted_coords_list = adjust_coordinates(coords_list, image_offsets)
    draw = ImageDraw.Draw(combined_image)

    num_pairs = len(all_matches)
    colors = generate_colors(num_pairs)

    for idx, (img_idx1, img_idx2, matches) in enumerate(all_matches):
        color = colors[idx]
        coords1 = adjusted_coords_list[img_idx1]
        coords2 = adjusted_coords_list[img_idx2]
        for idx1, idx2 in matches:
            y1, x1 = coords1[:, idx1]
            y2, x2 = coords2[:, idx2]
            draw.line([(x1, y1), (x2, y2)], fill=color, width=1)

    combined_image.save(output_path)
    print(f"All matches visualization saved to {output_path}.")


def main():
    if len(sys.argv) < 2:
        print("Error: Section argument is required.")
        sys.exit(1)

    section = sys.argv[1]
    # section = '12'

    images_dir = f'./images/{section}/'
    step3_dir = f'./part2_output/{section}/step3/'
    output_dir = f'./part2_output/{section}/step4/'
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    num_images = len(image_files)
    print(f"Found {num_images} images in {images_dir}.")

    descriptors_list = []
    coords_list = []
    for image_name in image_files:
        descriptors_path = os.path.join(step3_dir, f'descriptors_{image_name}.npy')
        coords_path = os.path.join(step3_dir, f'valid_coords_{image_name}.npy')
        descriptors = np.load(descriptors_path)
        coords = np.load(coords_path)
        descriptors_list.append(descriptors)
        coords_list.append(coords)
        print(f"Loaded descriptors and coordinates for {image_name}.")

    all_matches = match_all_images(image_files, descriptors_list, coords_list, output_dir)

    visualization_path = os.path.join(output_dir, 'all_matches.png')
    visualize_all_matches(image_paths, coords_list, all_matches, visualization_path)

    print("Feature matching and visualization completed.")


if __name__ == "__main__":
    main()
