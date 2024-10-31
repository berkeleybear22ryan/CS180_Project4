# b5_1.py
# computing homographies using RANSAC to align images

import os
import sys
import numpy as np
from PIL import Image, ImageDraw


def compute_homography(points1, points2):
    N = points1.shape[0]
    A = []
    for i in range(N):
        x, y = points1[i]
        x_prime, y_prime = points2[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]
    return H


def ransac_homography(matches, coords1, coords2, num_iterations=10000, error_threshold=1.0):
    max_inliers = []
    best_H = None

    points1 = np.array([coords1[:, idx1] for idx1, _ in matches])
    points2 = np.array([coords2[:, idx2] for _, idx2 in matches])

    N = len(matches)
    print(f"num_iterations: {num_iterations}")
    print(f"error_threshold: {error_threshold}")
    for _ in range(num_iterations):
        idx = np.random.choice(N, 4, replace=False)
        sample_points1 = points1[idx]
        sample_points2 = points2[idx]

        H = compute_homography(sample_points1, sample_points2)

        points1_homogeneous = np.hstack((points1, np.ones((N, 1))))
        projected_points = (H @ points1_homogeneous.T).T
        projected_points /= projected_points[:, [2]]
        projected_points = projected_points[:, :2]

        errors = np.linalg.norm(projected_points - points2, axis=1)

        inlier_mask = errors < error_threshold
        inliers = [matches[i] for i in range(N) if inlier_mask[i]]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_H = H

        if len(max_inliers) > 0.8 * N:
            print("early exit ...")
            break

    print(f"RANSAC: Found best model with {len(max_inliers)} inliers out of {N} correspondences.")
    return best_H, max_inliers


def visualize_inliers(image1_path, image2_path, coords1, coords2, inliers, output_path):
    print("Visualizing inlier matches...")
    im1 = Image.open(image1_path).convert('RGB')
    im2 = Image.open(image2_path).convert('RGB')

    width1, height1 = im1.size
    width2, height2 = im2.size
    total_width = width1 + width2
    max_height = max(height1, height2)
    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste(im1, (0, 0))
    new_im.paste(im2, (width1, 0))

    draw = ImageDraw.Draw(new_im)

    for idx1, idx2 in inliers:
        y1, x1 = coords1[:, idx1]
        y2, x2 = coords2[:, idx2]
        x2_adjusted = x2 + width1
        draw.line([(x1, y1), (x2_adjusted, y2)], fill=(0, 255, 0), width=2)
        r = 3
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=(255, 0, 0))
        draw.ellipse((x2_adjusted - r, y2 - r, x2_adjusted + r, y2 + r), outline=(255, 0, 0))

    new_im.save(output_path)
    print(f"Inlier matches visualization saved to {output_path}.")


def main():
    if len(sys.argv) < 2:
        print("Error: Section argument is required.")
        sys.exit(1)

    section = sys.argv[1]
    # section = '12'

    images_dir = f'./images/{section}/'
    step4_dir = f'./part2_output/{section}/step4/'
    step3_dir = f'./part2_output/{section}/step3/'
    output_dir = f'./part2_output/{section}/step5/'
    os.makedirs(output_dir, exist_ok=True)

    points_base_dir = f'./part2_output/{section}/points/'

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    num_images = len(image_files)
    print(f"Found {num_images} images in {images_dir}.")

    for i in range(num_images - 1):
        image1_name = image_files[i]
        image2_name = image_files[i + 1]
        print(f"\nProcessing images: {image1_name}, {image2_name}")

        matches_filename = f'matches_{image1_name}_{image2_name}.npy'
        matches_path = os.path.join(step4_dir, matches_filename)
        matches = np.load(matches_path, allow_pickle=True)
        matches = matches.tolist()

        N = len(matches)
        if N < 4:
            print(f"Not enough matches ({N}) between {image1_name} and {image2_name} to compute homography.")
            continue

        coords1_path = os.path.join(step3_dir, f'valid_coords_{image1_name}.npy')
        coords2_path = os.path.join(step3_dir, f'valid_coords_{image2_name}.npy')
        coords1 = np.load(coords1_path)
        coords2 = np.load(coords2_path)

        best_H, best_inliers = ransac_homography(matches, coords1, coords2)

        H_filename = os.path.join(output_dir, f'homography_{image1_name}_{image2_name}.npy')
        inliers_filename = os.path.join(output_dir, f'inliers_{image1_name}_{image2_name}.npy')
        np.save(H_filename, best_H)
        np.save(inliers_filename, best_inliers)
        print(f"Saved homography to {H_filename} and inliers to {inliers_filename}.")

        image1_path = os.path.join(images_dir, image1_name)
        image2_path = os.path.join(images_dir, image2_name)
        visualization_path = os.path.join(output_dir, f'inliers_{image1_name}_{image2_name}.png')
        visualize_inliers(image1_path, image2_path, coords1, coords2, best_inliers, visualization_path)

        image1_base = os.path.splitext(image1_name)[0]
        image2_base = os.path.splitext(image2_name)[0]

        points_dir = os.path.join(points_base_dir, f'{image1_base}_{image2_base}')
        os.makedirs(points_dir, exist_ok=True)

        points1_inliers = np.array([[coords1[1, idx1], coords1[0, idx1]] for idx1, _ in best_inliers])
        points2_inliers = np.array([[coords2[1, idx2], coords2[0, idx2]] for _, idx2 in best_inliers])

        pts1_filename = os.path.join(points_dir, f'{image1_base}_pts.txt')
        pts2_filename = os.path.join(points_dir, f'{image2_base}_pts.txt')

        np.savetxt(pts1_filename, points1_inliers, fmt='%.6f', delimiter=' ')
        np.savetxt(pts2_filename, points2_inliers, fmt='%.6f', delimiter=' ')

        print(f"Saved point correspondences to {pts1_filename} and {pts2_filename}.")

    print("RANSAC homography estimation and point correspondences saving completed.")


if __name__ == '__main__':
    main()
