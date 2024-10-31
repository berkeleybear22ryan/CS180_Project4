# THIS FILE is for projecting

import cv2
import numpy as np
import os
import sys


def warpImage(im, H, output_shape):
    h_out, w_out = output_shape
    h_im, w_im, c = im.shape

    x_out, y_out = np.meshgrid(np.arange(w_out), np.arange(h_out))
    x_out_flat = x_out.flatten()
    y_out_flat = y_out.flatten()
    ones = np.ones_like(x_out_flat)
    coords_out = np.vstack([x_out_flat, y_out_flat, ones])

    H_inv = np.linalg.inv(H)
    coords_in = H_inv @ coords_out
    x_in = coords_in[0, :] / coords_in[2, :]
    y_in = coords_in[1, :] / coords_in[2, :]

    im_warped = np.zeros((h_out, w_out, c), dtype=im.dtype)

    for i in range(c):
        print(f"{i} -- warpimage")
        channel = im[:, :, i]

        x0 = np.floor(x_in).astype(int)
        y0 = np.floor(y_in).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        dx = x_in - x0
        dy = y_in - y0

        valid = (x0 >= 0) & (x1 < w_im) & (y0 >= 0) & (y1 < h_im)

        im_channel = np.zeros_like(x_in)

        x0_valid = x0[valid]
        x1_valid = x1[valid]
        y0_valid = y0[valid]
        y1_valid = y1[valid]
        dx_valid = dx[valid]
        dy_valid = dy[valid]

        I00 = channel[y0_valid, x0_valid]
        I01 = channel[y1_valid, x0_valid]
        I10 = channel[y0_valid, x1_valid]
        I11 = channel[y1_valid, x1_valid]

        im_channel[valid] = (I00 * (1 - dx_valid) * (1 - dy_valid) +
                             I10 * dx_valid * (1 - dy_valid) +
                             I01 * (1 - dx_valid) * dy_valid +
                             I11 * dx_valid * dy_valid)

        im_warped[:, :, i] = im_channel.reshape(h_out, w_out)

    return im_warped


def blend_images(mosaic, im_warped, mask_warped):
    h, w, c = mosaic.shape

    for i in range(c):
        print(f"{i} -- blendimage")
        mosaic_channel = mosaic[:, :, i]
        im_channel = im_warped[:, :, i]

        overlap = (mosaic_channel > 0) & (mask_warped > 0)

        non_overlap = (mosaic_channel == 0) & (mask_warped > 0)

        mosaic_channel[overlap] = (mosaic_channel[overlap] * 0.5 + im_channel[overlap] * 0.5)

        mosaic_channel[non_overlap] = im_channel[non_overlap]

        mosaic[:, :, i] = mosaic_channel

    return mosaic


def main():

    if len(sys.argv) < 2:
        print("Error: Section argument is required.")
        sys.exit(1)

    dir_name = sys.argv[1]
    # dir_name = "12"

    # ... directories
    image_dir = f'./images/{dir_name}/'
    h_matrix_dir = f'./h_matrix/{dir_name}/'
    output_dir = f'./output/{dir_name}/'
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    images = [cv2.imread(os.path.join(image_dir, f)) for f in image_files]

    base_image_name = os.path.splitext(image_files[0])[0]

    H_list = []
    H_identity = np.eye(3)
    H_list.append(H_identity)

    for i in range(1, len(images)):
        image_name = os.path.splitext(image_files[i])[0]
        h_matrix_filename = f"H_{image_name}_to_{base_image_name}.txt"
        h_matrix_filepath = os.path.join(h_matrix_dir, h_matrix_filename)
        if os.path.exists(h_matrix_filepath):
            H = np.loadtxt(h_matrix_filepath)
            H_list.append(H)
        else:
            print(f"Homography file {h_matrix_filepath} not found.")
            return

    corners = []
    for i, H in enumerate(H_list):
        h_im, w_im = images[i].shape[:2]
        corners_im = np.array([[0, 0], [w_im, 0], [w_im, h_im], [0, h_im]])
        corners_im_homog = np.hstack([corners_im, np.ones((4, 1))])
        warped_corners = (H @ corners_im_homog.T).T
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2, np.newaxis]
        corners.append(warped_corners)

    corners = np.vstack(corners)
    x_min, y_min = np.int32(corners.min(axis=0))
    x_max, y_max = np.int32(corners.max(axis=0))
    offset_x, offset_y = -x_min, -y_min
    mosaic_width = x_max - x_min
    mosaic_height = y_max - y_min

    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

    for i, H in enumerate(H_list):
        print(i)
        im = images[i]
        H_offset = np.eye(3)
        H_offset[0, 2] = offset_x
        H_offset[1, 2] = offset_y
        H_warp = H_offset @ H

        im_warped = warpImage(im, H_warp, (mosaic_height, mosaic_width))
        warped_image_filename = f"warped_{os.path.splitext(image_files[i])[0]}.jpg"
        cv2.imwrite(os.path.join(output_dir, warped_image_filename), im_warped)
        print(f"Warped image saved: {warped_image_filename}")

        mask_warped = np.any(im_warped > 0, axis=2).astype(np.uint8)

        mosaic = blend_images(mosaic, im_warped, mask_warped)

    mosaic_filename = 'mosaic.jpg'
    cv2.imwrite(os.path.join(output_dir, mosaic_filename), mosaic)
    print(f"Mosaic saved at {os.path.join(output_dir, mosaic_filename)}")


if __name__ == '__main__':
    main()
