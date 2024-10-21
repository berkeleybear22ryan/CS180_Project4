# THIS FILE is for getting the H's
import numpy as np
import os


# TODO: keep there naming ...
def computeH(im1_pts, im2_pts):
    n = im1_pts.shape[0]
    if n < 4:
        raise ValueError("At least 4 point correspondences are required.")

    A = []
    for i in range(n):
        x, y = im2_pts[i]
        x_p, y_p = im1_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_p, y * x_p, x_p])
        A.append([0, 0, 0, -x, -y, -1, x * y_p, y * y_p, y_p])
    A = np.array(A)

    U, S, Vh = np.linalg.svd(A)
    h = Vh[-1]
    H = h.reshape((3, 3))
    H = H / H[2, 2]

    return H


def main():
    dir_name = "9_7"
    image_dir = f'./images/{dir_name}/'
    points_dir = f'./points/{dir_name}/'
    h_matrix_dir = f'./h_matrix/{dir_name}/'
    os.makedirs(h_matrix_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    num_images = len(image_files)

    H_list = []

    H_identity = np.eye(3)
    H_list.append(H_identity)

    for i in range(num_images - 1):
        im1_name = os.path.splitext(image_files[i])[0]
        im2_name = os.path.splitext(image_files[i + 1])[0]

        pair_dir = os.path.join(points_dir, f"{im1_name}_{im2_name}")

        im1_pts = np.loadtxt(os.path.join(pair_dir, f"{im1_name}_pts.txt"))
        im2_pts = np.loadtxt(os.path.join(pair_dir, f"{im2_name}_pts.txt"))

        H_i1_i = computeH(im1_pts, im2_pts)

        H_i1_0 = H_list[i] @ H_i1_i

        H_list.append(H_i1_0)

        h_matrix_name = f"H_{im2_name}_to_{image_files[0][:-4]}.txt"
        np.savetxt(os.path.join(h_matrix_dir, h_matrix_name), H_i1_0)

        print(f"Computed and saved homography for {im2_name} to base image")


if __name__ == '__main__':
    main()
