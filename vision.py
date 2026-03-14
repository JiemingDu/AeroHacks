# pylint: disable=no-member
import cv2
import numpy as np


# ==========================================
# CAMERA HELPERS
# ==========================================

def open_cameras(front_index=1, side_index=2):
    """
    Opens front and side USB cameras.

    Args:
        front_index : device index for front camera (default 1)
        side_index  : device index for side camera  (default 2)

    Returns:
        (cam_front, cam_side) VideoCapture objects

    Raises:
        RuntimeError if either camera fails to open
    """
    cam_front = cv2.VideoCapture(front_index)
    cam_side  = cv2.VideoCapture(side_index)

    if not cam_front.isOpened():
        raise RuntimeError(f"Front camera (index {front_index}) failed to open.")
    if not cam_side.isOpened():
        raise RuntimeError(f"Side camera (index {side_index}) failed to open.")

    print(f"Front camera opened (index {front_index})")
    print(f"Side camera opened  (index {side_index})")
    return cam_front, cam_side


def release_cameras(*cams):
    """Releases all provided VideoCapture objects and destroys OpenCV windows."""
    for cam in cams:
        cam.release()
    cv2.destroyAllWindows()
    print("Cameras released.")


# ==========================================
# LED DETECTION
# ==========================================

def get_drone_pixel_position(frame, prev_pos=None, search_radius=None):
    """
    Finds the green LED in a frame and returns its (x, y) pixel centroid.

    The green LED is the tracking target. Uses HSV masking + largest contour.

    Args:
        frame         : BGR image from cv2.VideoCapture.read()
        prev_pos      : optional (x, y) of last known position. If provided
                        with search_radius, only a square ROI around this
                        point is searched.
        search_radius : half-size of ROI box in pixels (int). Ignored if
                        prev_pos is None.

    Returns:
        (cx, cy) pixel coordinates, or (None, None) if not found
    """
    h, w = frame.shape[:2]

    if prev_pos is not None and search_radius is not None:
        px, py = prev_pos
        x1 = max(0, px - search_radius)
        y1 = max(0, py - search_radius)
        x2 = min(w, px + search_radius)
        y2 = min(h, py + search_radius)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None
    else:
        x1, y1 = 0, 0
        roi = frame

    blurred = cv2.GaussianBlur(roi, (11, 11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # tune these HSV bounds to your lighting conditions
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    mask      = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)

    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"]) + x1
    cy = int(M["m01"] / M["m00"]) + y1
    return cx, cy