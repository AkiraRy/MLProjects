import cv2
import numpy as np
import os

IMAGE_DIR = r'.\images'
WINDOW_NAME = "Images"
MAX_DIM = 800
BLUR_KERNEL_SIZE = 6


# steps
# drawing plate
# 1. blur
# 2. bw
# 3. edges
# 4.　_find_counters + draw_plate_edge

# finding circles
# 1. bw
# 2. blur
# 3. circles


def process_image_get_counters(image):
    blurred = apply_blur(image, ksize=11)
    img_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = detect_edges(img_gray)
    return _find_contours(edges)


def process_image_get_circles(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = apply_blur(img_gray)
    return get_circles(blurred)


def get_circles(preprocessed_image):
    circles = cv2.HoughCircles(preprocessed_image, cv2.HOUGH_GRADIENT, 1, 13, param1=69, param2=30, minRadius=18,
                               maxRadius=35)
    return np.uint16(np.around(circles)) if circles is not None else []


def display_results(on_plate, off_plate):
    sum_on_plate = on_plate['big'] *5 + on_plate['small'] * 0.05
    sum_off_plate = off_plate['big'] *5 + off_plate['small'] * 0.05

    print(f"Amount of big coins on the plate: {on_plate['big']}")
    print(f"Amount of small coins on the plate:  {on_plate['small']}")
    print(f"Amount of big coins off the plate: {off_plate['big']}")
    print(f"Amount of big coins off the plate: {off_plate['small']}")
    print(f"Total sum of money on the plate {sum_on_plate}")
    print(f"Total sum of money off the plate {sum_off_plate}")
    print(f"Total sum of money: {sum_on_plate+sum_off_plate}")
    print(f"====================================================")


def draw_circles(circles, image, plate_rect=None):
    if len(circles) == 0:
        print("No circles to draw.")
        return image

    coin_stats = {
        "on_plate": {"big": 0, "small": 0},
        "off_plate": {"big": 0, "small": 0}
    }

    for circle in circles[0, :]:
        x, y, r = circle
        area = np.pi * (r**2)

        on_plate = False
        if plate_rect:
            px, py, pw, ph = plate_rect
            if px <= x <= px + pw and py <= y <= py + ph:
                on_plate = True

        if r > 26:
            color = (0, 255, 0)
            size = "big"
        else:
            color = (255, 255, 0)
            size = "small"

        cv2.circle(image, (x, y), r, color, 2)
        if on_plate:
            coin_stats["on_plate"][size] += 1
        else:
            coin_stats["off_plate"][size] += 1

    coins_on_plate = coin_stats["on_plate"]
    coins_off_plate = coin_stats["off_plate"]
    return image, coins_on_plate, coins_off_plate


def _find_contours(preprocessed_image):
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def draw_plate_edge(contours, image):
    x, y, w, h = cv2.boundingRect(contours)
    return cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1), w * h


def apply_blur(image, blur_type="median", ksize=None):
    if not ksize:
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
    return cv2.Canny(image, 50, 150)


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
    cv2.createTrackbar('KSize', WINDOW_NAME, 5, 50, lambda x: None)
    cv2.imshow(WINDOW_NAME, current_image)

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF

        if ord('0') <= key <= ord('9'):
            index = key - ord('0')
            if index > len(files) - 1:
                continue
            current_index = index
            current_image = load_image(current_index, files)
            original_image = current_image.copy()
        elif key == ord('o'):
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
        elif key == ord('c'):
            circles = get_circles(current_image)
            current_image = original_image.copy()
            current_image = draw_circles(circles, current_image)
        elif key == ord("a"):
            mask = current_image
            if current_image.ndim == 3:
                mask = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            current_image = bitwise(original_image, mask=mask)

        elif key == ord("t"):
            contours = process_image_get_counters(current_image)
            circles = process_image_get_circles(current_image)
            plate_rect = None
            if contours is not None:
                plate_rect = cv2.boundingRect(contours)

            current_image, coins_on_plate, coins_off_plate = draw_circles(circles, current_image, plate_rect=plate_rect)
            display_results(coins_on_plate, coins_off_plate)

            if plate_rect:
                current_image, area_plate = draw_plate_edge(contours, current_image)

        elif key == 27:
            cv2.destroyAllWindows()
            break

        cv2.imshow(WINDOW_NAME, current_image)


if __name__ == '__main__':
    main()
