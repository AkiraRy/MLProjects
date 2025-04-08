import cv2
import numpy as np
import os

IMAGE_DIR = r'.\images'
WINDOW_NAME = "Images"
MAX_DIM = 800
BLUR_KERNEL_SIZE = 13
# steps
# 1. bw
# 2. blur
# 3. edges
# 4.　_find_counters + draw_plate_edge


def _find_contours(preprocessed_image):
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def draw_plate_edge(contours, image):
    x, y, w, h = cv2.boundingRect(contours)
    return cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2), w*h


def apply_blur(image, blur_type="median"):
    ksize = cv2.getTrackbarPos('KSize', WINDOW_NAME) or 1
    ksize = max(1, ksize | 1)
    if blur_type == "median":
        return cv2.medianBlur(image, ksize)
    return cv2.blur(image, (ksize, ksize))


def detect_edges(image, edge_type="Canny"):
    ksize = cv2.getTrackbarPos('KSize', WINDOW_NAME) or 1
    ksize = max(1, ksize | 1)
    if edge_type == "Laplacian":
        lap = cv2.Laplacian(image, ksize=ksize, ddepth=cv2.CV_64F)
        _, thresholded_laplacian = cv2.threshold(lap, 50, 255, cv2.THRESH_BINARY)
        return thresholded_laplacian
    return cv2.Canny(image, 55, 30)


def bitwise(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def resize_image(image, max_dim=MAX_DIM):
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def load_image(index, files):
    filepath = os.path.join(IMAGE_DIR, files[index])
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Could not read image at {filepath}")
    return resize_image(image)


def main():
    files = os.listdir(IMAGE_DIR)
    if not files:
        print("No files found in the specified directory.")
        return

    current_index = 0

    current_image = load_image(current_index, files)
    original_image = current_image.copy()

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar('KSize', WINDOW_NAME, 10, 50, lambda x: None)
    cv2.imshow(WINDOW_NAME, current_image)

    action = None
    area_plate = 0
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF

        if ord('0') <= key <= ord('9'):
            index = key - ord('0')
            if index > len(files)-1:
                continue

            current_index = index
            current_image = load_image(current_index, files)
            original_image = current_image.copy()
        elif key == ord('o'):
            action=''
            current_image = original_image.copy()
        elif key == ord('b'):
            if current_image.ndim != 3:
                continue
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        elif key == ord('d'):
            current_image = apply_blur(current_image)
        elif key == ord('e'):
            current_image = detect_edges(current_image)
        elif key == ord('p'):
            contours = _find_contours(current_image)
            current_image = original_image.copy()
            current_image, area_plate = draw_plate_edge(contours, current_image)
            print(area_plate)
        elif key == ord("a"):
            mask = current_image
            if current_image.ndim == 3:
                mask = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            current_image = bitwise(original_image, mask=mask)
        elif key == 27:
            cv2.destroyAllWindows()
            break

        cv2.imshow(WINDOW_NAME, current_image)


if __name__ == '__main__':
    main()


# todo
# have binds for each single action
# have a single function on some key which will do all the job at once

