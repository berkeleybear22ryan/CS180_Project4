# NOTE: need to fix issue with sometimes auto clicks a point ... do later ... has to do with resolution I think
# THIS FILE is for creating the points
import cv2
import numpy as np
import os
import sys


def click_event(event, x, y, flags, params):
    pts1, pts2, img, win_name, img_copy, divider, scale_factors = params
    scale_factor1, scale_factor2 = scale_factors
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < divider:
            x1, y1 = int(x / scale_factor1), int(y / scale_factor1)
            pts1.append([x1, y1])
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img, str(len(pts1)), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            x2, y2 = int((x - divider) / scale_factor2), int(y / scale_factor2)
            pts2.append([x2, y2])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, str(len(pts2)), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win_name, img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if x < divider:
            if pts1:
                pts1.pop()
                img[:] = img_copy.copy()
                redraw_points(img, pts1, pts2, divider, scale_factors)
                cv2.imshow(win_name, img)
        else:
            if pts2:
                pts2.pop()
                img[:] = img_copy.copy()
                redraw_points(img, pts1, pts2, divider, scale_factors)
                cv2.imshow(win_name, img)


def redraw_points(img, pts1, pts2, divider, scale_factors):
    scale_factor1, scale_factor2 = scale_factors
    for i, pt in enumerate(pts1):
        x = int(pt[0] * scale_factor1)
        y = int(pt[1] * scale_factor1)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(i + 1), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, pt in enumerate(pts2):
        x = int(pt[0] * scale_factor2) + divider
        y = int(pt[1] * scale_factor2)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)


def load_existing_points(pts1, pts2, img, divider, scale_factors):
    scale_factor1, scale_factor2 = scale_factors
    for i, pt in enumerate(pts1):
        x = int(pt[0] * scale_factor1)
        y = int(pt[1] * scale_factor1)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(i + 1), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, pt in enumerate(pts2):
        x = int(pt[0] * scale_factor2) + divider
        y = int(pt[1] * scale_factor2)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)


def select_correspondences_cv2(image_pairs, current_pair_index, output_dir, add=False):
    total_pairs = len(image_pairs)
    while True:
        if current_pair_index < 0 or current_pair_index >= total_pairs:
            print("No more image pairs in this direction.")
            break

        (im1, im1_name), (im2, im2_name) = image_pairs[current_pair_index]

        pts1 = []
        pts2 = []

        pair_dir = os.path.join(output_dir, f"{im1_name}_{im2_name}")
        if add:
            try:
                pts1 = np.loadtxt(os.path.join(pair_dir, f"{im1_name}_pts.txt"))
                pts2 = np.loadtxt(os.path.join(pair_dir, f"{im2_name}_pts.txt"))

                if pts1.ndim == 1:
                    pts1 = pts1.reshape(-1, 2)
                if pts2.ndim == 1:
                    pts2 = pts2.reshape(-1, 2)

                pts1 = pts1.tolist()
                pts2 = pts2.tolist()

                print(f"Loaded existing points for {im1_name} and {im2_name}")
            except Exception as e:
                print(f"No existing points found for {im1_name} and {im2_name}: {e}")

        # ... check this
        max_width = 2000  # FIXME: max width for display
        scale_factor1 = min(1.0, max_width / im1.shape[1])
        scale_factor2 = min(1.0, max_width / im2.shape[1])

        im1_display = cv2.resize(im1, None, fx=scale_factor1, fy=scale_factor1, interpolation=cv2.INTER_AREA)
        im2_display = cv2.resize(im2, None, fx=scale_factor2, fy=scale_factor2, interpolation=cv2.INTER_AREA)

        h1, w1 = im1_display.shape[:2]
        h2, w2 = im2_display.shape[:2]
        max_height = max(h1, h2)
        total_width = w1 + w2
        combined_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        combined_image[:h1, :w1] = im1_display
        combined_image[:h2, w1:w1 + w2] = im2_display
        img_copy = combined_image.copy()
        divider = w1

        win_name = f'Image Pair: {im1_name} & {im2_name}'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1000, 600)
        load_existing_points(pts1, pts2, combined_image, divider, (scale_factor1, scale_factor2))
        cv2.setMouseCallback(win_name, click_event,
                             [pts1, pts2, combined_image, win_name, img_copy, divider, (scale_factor1, scale_factor2)])
        print(f"Select points on images. Use 'a' and 'd' to navigate, right-click to undo.")

        while True:
            cv2.imshow(win_name, combined_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                if len(pts1) != len(pts2):
                    print("Number of points in both images must be the same.")
                    continue
                save_points(pts1, pts2, im1_name, im2_name, output_dir)
                current_pair_index -= 1
                cv2.destroyWindow(win_name)
                return current_pair_index
            elif key == ord('d'):
                if len(pts1) != len(pts2):
                    print("Number of points in both images must be the same.")
                    continue
                save_points(pts1, pts2, im1_name, im2_name, output_dir)
                current_pair_index += 1
                cv2.destroyWindow(win_name)
                return current_pair_index
            elif key == ord('q'):
                cv2.destroyWindow(win_name)
                sys.exit()
    return current_pair_index


def save_points(pts1, pts2, im1_name, im2_name, output_dir):
    pts1_array = np.array(pts1)
    pts2_array = np.array(pts2)

    pair_dir = os.path.join(output_dir, f"{im1_name}_{im2_name}")
    os.makedirs(pair_dir, exist_ok=True)
    np.savetxt(os.path.join(pair_dir, f"{im1_name}_pts.txt"), pts1_array)
    np.savetxt(os.path.join(pair_dir, f"{im2_name}_pts.txt"), pts2_array)
    print(f"Saved points for {im1_name} and {im2_name} in {pair_dir}")


def main():
    dir_name = "10_0_P2"
    image_dir = f'./images/{dir_name}/'

    # part1
    output_dir = f'./points/{dir_name}/'
    # part2
    # output_dir = f'./part2_output/{dir_name}/points'

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    images = [cv2.imread(os.path.join(image_dir, f)) for f in image_files]
    print(image_files)

    num_images = len(images)

    image_pairs = []
    for i in range(num_images - 1):
        im1 = images[i]
        im2 = images[i + 1]
        im1_name = os.path.splitext(image_files[i])[0]
        im2_name = os.path.splitext(image_files[i + 1])[0]
        image_pairs.append(((im1, im1_name), (im2, im2_name)))

    add_mode = '-add' in sys.argv  # ... -add flag in command line ... check this

    current_pair_index = 0
    while 0 <= current_pair_index < len(image_pairs):
        current_pair_index = select_correspondences_cv2(
            image_pairs, current_pair_index, output_dir, add=add_mode
        )
        if current_pair_index < 0 or current_pair_index >= len(image_pairs):
            print("No more image pairs to process.")
            break


if __name__ == '__main__':
    main()
