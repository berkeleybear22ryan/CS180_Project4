# THIS FILE is for rectifying ...
import cv2
import numpy as np


def click_event(event, x, y, flags, params):
    pts, img, win_name = params
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(len(pts)), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win_name, img)


def select_points_for_rectification(image_path):
    img = cv2.imread(image_path)
    img_display = img.copy()
    win_name = "Select Points for Rectification"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img_display)

    pts = []
    cv2.setMouseCallback(win_name, click_event, [pts, img_display, win_name])

    print("Click on the four corners of the object to rectify. Close the window when done.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(pts, dtype='float32')


def rectify_image(image, pts_src):
    h, w = image.shape[:2]

    x_min, y_min = np.min(pts_src, axis=0)
    x_max, y_max = np.max(pts_src, axis=0)
    pts_dst = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_min, y_max],
        [x_max, y_max]
    ], dtype='float32')

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[3] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[2] = pts[np.argmax(diff)]
        return rect

    pts_src_ordered = order_points(pts_src)
    pts_dst_ordered = order_points(pts_dst)

    H, status = cv2.findHomography(pts_src_ordered, pts_dst_ordered)

    rectified_image = cv2.warpPerspective(image, H, (w, h))

    return rectified_image


def main():
    name = "7"
    mosaic_image_path = f'./finals/{name}.jpg'
    mosaic_image = cv2.imread(mosaic_image_path)

    pts_src = select_points_for_rectification(mosaic_image_path)

    if len(pts_src) != 4:
        print("You must select exactly four points.")
        return

    rectified_image = rectify_image(mosaic_image, pts_src)

    cv2.namedWindow('Rectified Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Rectified Image', rectified_image)
    cv2.imwrite(f'{name}_rectified_image.jpg', rectified_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
