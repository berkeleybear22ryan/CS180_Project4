# THIS FILE is extra beyond requirments that adds to a3.py which is for projecting but adds more smoothing and shapening and stuff like that b/c no time to select better points
# ... will do auto points in part 2 anyways ... just had to do a lot of point selecting and was tired of trying to make it perfect b/c lot of time wasted
import cv2
import numpy as np
import os
# assume this was okay as it was not even required in project
from scipy.ndimage import distance_transform_edt


def warpImage(im, H, output_shape):
    h_out, w_out = output_shape
    h_im, w_im, c = im.shape

    print(f"Warping image of size {h_im}x{w_im} to size {h_out}x{w_out}")

    y_out, x_out = np.indices((h_out, w_out))
    ones = np.ones_like(x_out)
    coords_out = np.stack([x_out, y_out, ones], axis=-1).reshape(-1, 3).T  # Shape: (3, h_out * w_out)

    H_inv = np.linalg.inv(H)
    coords_in = H_inv @ coords_out
    coords_in /= coords_in[2, :]

    x_in = coords_in[0, :]
    y_in = coords_in[1, :]

    im_warped = np.zeros((h_out * w_out, c), dtype=im.dtype)

    x_in = x_in.reshape(-1)
    y_in = y_in.reshape(-1)

    x0 = np.floor(x_in).astype(int)
    y0 = np.floor(y_in).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0_clipped = np.clip(x0, 0, w_im - 1)
    y0_clipped = np.clip(y0, 0, h_im - 1)
    x1_clipped = np.clip(x1, 0, w_im - 1)
    y1_clipped = np.clip(y1, 0, h_im - 1)

    wa = (x1 - x_in) * (y1 - y_in)
    wb = (x1 - x_in) * (y_in - y0)
    wc = (x_in - x0) * (y1 - y_in)
    wd = (x_in - x0) * (y_in - y0)

    valid_mask = (x0 >= 0) & (x1 < w_im) & (y0 >= 0) & (y1 < h_im)

    for i in range(c):
        print(f"Processing channel {i + 1}/{c} in warpImage")
        channel = im[:, :, i]

        Ia = channel[y0_clipped, x0_clipped]
        Ib = channel[y1_clipped, x0_clipped]
        Ic = channel[y0_clipped, x1_clipped]
        Id = channel[y1_clipped, x1_clipped]

        Iw = wa * Ia + wb * Ib + wc * Ic + wd * Id

        im_warped[valid_mask, i] = Iw[valid_mask]

    im_warped = im_warped.reshape(h_out, w_out, c)
    return im_warped


def blend_images(mosaic, im_warped):
    print("Blending images")
    mask_mosaic = np.any(mosaic > 0, axis=2).astype(np.float32)
    mask_warped = np.any(im_warped > 0, axis=2).astype(np.float32)

    overlap = (mask_mosaic > 0) & (mask_warped > 0)
    mosaic_only = (mask_mosaic > 0) & (mask_warped == 0)
    warped_only = (mask_mosaic == 0) & (mask_warped > 0)

    alpha = np.zeros_like(mask_mosaic)

    dist_mosaic = distance_transform_edt(mask_mosaic)
    dist_warped = distance_transform_edt(mask_warped)

    total_dist = dist_mosaic + dist_warped + 1e-6

    alpha = dist_warped / total_dist

    alpha = np.clip(alpha, 0, 1)

    alpha = alpha[:, :, np.newaxis]  # Add channel dimension
    mosaic[overlap] = mosaic[overlap] * (1 - alpha[overlap]) + im_warped[overlap] * alpha[overlap]
    mosaic[warped_only] = im_warped[warped_only]

    return mosaic


def sharpen_image(image):
    print("Sharpening image")
    blurred = gaussian_blur(image, kernel_size=5, sigma=1)

    sharpened = image + (image - blurred) * 1.0
    sharpened = np.clip(sharpened, 0, 255)
    return sharpened


def gaussian_blur(image, kernel_size=5, sigma=1):
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    kernel_1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / np.sum(kernel_1d)

    blurred = convolve_separable(image, kernel_1d)
    return blurred


def convolve_separable(image, kernel_1d):
    temp = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=1, arr=image)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=0, arr=temp)
    return blurred


def main():
    dir_name = "9_7"
    image_dir = f'./images/{dir_name}/'
    h_matrix_dir = f'./h_matrix/{dir_name}/'
    output_dir = f'./output/{dir_name}/'
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    images = [cv2.imread(os.path.join(image_dir, f)).astype(np.float32) for f in image_files]

    base_image_name = os.path.splitext(image_files[0])[0]

    H_list = []
    H_identity = np.eye(3)
    H_list.append(H_identity)

    for i in range(1, len(images)):
        image_name = os.path.splitext(image_files[i])[0]
        h_matrix_filename = f"H_{image_name}_to_{base_image_name}.txt"
        h_matrix_filepath = os.path.join(h_matrix_dir, h_matrix_filename)
        if os.path.exists(h_matrix_filepath):
            print(f"Loading homography {h_matrix_filename}")
            H = np.loadtxt(h_matrix_filepath)
            H_list.append(H)
        else:
            print(f"Homography file {h_matrix_filepath} not found.")
            return

    print("Determining mosaic size")
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
    print(f"Mosaic size will be {mosaic_width}x{mosaic_height}")

    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.float32)

    for i, H in enumerate(H_list):
        print(f"\nWarping and blending image {i + 1}/{len(images)}")
        im = images[i]
        H_offset = np.eye(3)
        H_offset[0, 2] = offset_x
        H_offset[1, 2] = offset_y
        H_warp = H_offset @ H

        im_warped = warpImage(im, H_warp, (mosaic_height, mosaic_width))

        warped_image_filename = f"warped_{os.path.splitext(image_files[i])[0]}.jpg"
        cv2.imwrite(os.path.join(output_dir, warped_image_filename), im_warped.astype(np.uint8))
        print(f"Warped image saved: {warped_image_filename}")

        mosaic = blend_images(mosaic, im_warped)

    mosaic = np.clip(mosaic, 0, 255)

    blended_filename = 'mosaic_blended.jpg'
    cv2.imwrite(os.path.join(output_dir, blended_filename), mosaic.astype(np.uint8))
    print(f"Blended mosaic saved at {os.path.join(output_dir, blended_filename)}")

    mosaic_sharpened = sharpen_image(mosaic)

    mosaic_sharpened_filename = 'mosaic_sharpened.jpg'
    cv2.imwrite(os.path.join(output_dir, mosaic_sharpened_filename), mosaic_sharpened.astype(np.uint8))
    print(f"Sharpened mosaic saved at {os.path.join(output_dir, mosaic_sharpened_filename)}")


if __name__ == '__main__':
    main()
