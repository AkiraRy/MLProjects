import cv2
import numpy as np
import os

IMAGE_DIR = r'.\images'
WINDOW_NAME = "Images"
MAX_DIM = 800


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
    image = load_image(current_index, files)
    original_image = image.copy()
    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, image)

    action = None
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF

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
        elif key == 27:
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    main()


# todo
# have binds for each single action
# have a single function on some key which will do all the job at once

