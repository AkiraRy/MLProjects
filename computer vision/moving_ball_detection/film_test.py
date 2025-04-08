import cv2
import numpy as np


def apply_hsv_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])  # not for red, but for my custom color.
    upper = np.array([255, 120, 120])  # mixture of gray and black?
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def apply_morphology(mask, ksize_open=5, ksize_close=5):
    kernel_open = np.ones((ksize_open, ksize_open), np.uint8)
    kernel_close = np.ones((ksize_close, ksize_close), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def draw_marker(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawMarker(image, (cx, cy), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)


def add_marker(frame):
    bitwise_mask = apply_hsv_filter(frame)
    morphology_close_mask = apply_morphology(bitwise_mask)
    draw_marker(frame, morphology_close_mask)


def main():
    input_path = './video2.mp4'
    output_path = 'result2.avi'

    # Open video file
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    size = (frame_width, frame_height)

    result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    frame_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break
        frame_count += 1

        # if frame_count % 2 == 0:
        add_marker(frame)

        result.write(frame)

    # Release resources
    video.release()
    result.release()
    print(f"Processed {frame_count} frames. Output saved to '{output_path}'.")


def rgb_to_hsv(r, g, b):
    rgb = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[0][0]


if __name__ == "__main__":
    main()
