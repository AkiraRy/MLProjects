import cv2
import numpy as np
import os

IMAGE_DIR = r'.\images'
WINDOW_NAME = "Images"
MAX_DIM = 800


# Helper functions
def load_image(index, files):
    filepath = os.path.join(IMAGE_DIR, files[index])
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Could not read image at {filepath}")
    return resize_image(image)


def resize_image(image, max_dim=MAX_DIM):
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def apply_hsv_filter(image, low_color, high_color, operation=None, ksize=5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 0, 0])
    upper = np.array([high_color, 120, 120])
    mask = cv2.inRange(hsv, lower, upper)

    if operation == 'bitwise':
        return cv2.bitwise_and(image, image, mask=mask)
    elif operation == 'median':
        return cv2.medianBlur(mask, ksize)
    elif operation == 'morphology_open':
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'morphology_close':
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def draw_marker(image, low_color, high_color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 0, 0])
    upper = np.array([high_color, 20, 20])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawMarker(image, (cx, cy), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=100, thickness=2)


def update_image(image, action):
    low_color = cv2.getTrackbarPos('Low', WINDOW_NAME)
    high_color = cv2.getTrackbarPos('High', WINDOW_NAME)
    ksize = cv2.getTrackbarPos('KSize', WINDOW_NAME) or 1
    ksize = max(1, ksize | 1)

    if action == 'hsv_range':
        output = apply_hsv_filter(image, low_color, high_color)
    elif action == 'bitwise':
        output = apply_hsv_filter(image, low_color, high_color, 'bitwise')
    elif action == 'median':
        output = apply_hsv_filter(image, low_color, high_color, 'wmedian', ksize)
    elif action == 'morphology_open':
        output = apply_hsv_filter(image, low_color, high_color, 'morphology_open', ksize)
    elif action == 'morphology_close':
        output = apply_hsv_filter(image, low_color, high_color, 'morphology_close', ksize)
    elif action == 'marker':
        output = image.copy()
        draw_marker(output, low_color, high_color)
    else:
        output = image

    cv2.imshow(WINDOW_NAME, output)


def main():
    files = os.listdir(IMAGE_DIR)
    if not files:
        print("No files found in the specified directory.")
        return

    current_index = 0
    image = load_image(current_index, files)
    original_image = image.copy()

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar('Low', WINDOW_NAME, 0, 255, lambda x: None)
    cv2.createTrackbar('High', WINDOW_NAME, 0, 255, lambda x: None)
    cv2.createTrackbar('KSize', WINDOW_NAME, 5, 50, lambda x: None)

    cv2.imshow(WINDOW_NAME, image)

    action = None
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF

        # choose Image
        if ord('0') <= key <= ord('9'):
            print(f"picture change")
            index = key - ord('0')
            if 0 <= index < len(files):
                current_index = index
                image = load_image(current_index, files)
                original_image = image.copy()
                cv2.imshow(WINDOW_NAME, image)
        elif key == ord('o'):
            print(f"reverse to original")
            action = ''
            image = original_image.copy()
            cv2.imshow(WINDOW_NAME, image)
        #  Resize
        elif key == ord('-') or key == ord('_'):
            print(f"downsize")
            if image.shape[0] > 10 and image.shape[1] > 10:
                image = cv2.resize(image, (int(image.shape[1] * 0.9), int(image.shape[0] * 0.9)),
                                   interpolation=cv2.INTER_LINEAR)
                cv2.imshow(WINDOW_NAME, image)
        elif key == ord('+') or key == ord('='):
            print(f"upsize")
            new_width = int(image.shape[1] * 1.1)
            new_height = int(image.shape[0] * 1.1)
            if new_width <= MAX_DIM and new_height <= MAX_DIM:
                image = cv2.resize(image, (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)
                cv2.imshow(WINDOW_NAME, image)
        #  Colors
        elif key == ord('q'):
            print(f"quit")
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        elif key == ord('w'):
            print(f"convert to hsv, and simply show it")
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        # === HSV Filtering and Operations ===
        elif key == ord('e'):

            action = 'hsv_range'
        elif key == ord('r'):
            action = 'bitwise'
        elif key == ord('t'):
            action = 'median'
        elif key == ord('f'):
            action = 'morphology_open'
        elif key == ord('g'):
            action = 'morphology_close'
        elif key == ord('h'):
            action = 'marker'
        # === Reset to Original Color ===

        # === HSV Channels ===
        elif key == ord('z'):
            # h = hue
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0])
        elif key == ord('x'):
            # s = saturation
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1])
        elif key == ord('c'):
            # v = value
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2])

        # === Filters ===
        elif key == ord('a'):
            cv2.imshow(WINDOW_NAME, cv2.Canny(image, 55, 30))
        elif key == ord('s'):
            cv2.imshow(WINDOW_NAME, cv2.blur(image, (7, 7)))
        elif key == ord('d'):
            blurred = cv2.blur(image, (7, 7))
            cv2.imshow(WINDOW_NAME, cv2.Canny(blurred, 55, 30))

        elif key == 27:
            cv2.destroyAllWindows()
            break

        if action:
            update_image(image, action)
            # action = None


if __name__ == '__main__':
    main()
