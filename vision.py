# pylint: disable=no-member
import cv2
import numpy as np


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_drone_pixel_position(frame, prev_pos=None, search_radius=100):
    """
    Searches for Green, Blue, and Red LEDs.
    Averages the positions of whichever ones it finds to get the true drone center.
    Limits search to 'search_radius' if prev_pos is known.
    """
    h, w = frame.shape[:2]

    # --- ROI (Region of Interest) Logic ---
    if prev_pos is not None:
        px, py = prev_pos
        x_min = max(0, px - search_radius)
        y_min = max(0, py - search_radius)
        x_max = min(w, px + search_radius)
        y_max = min(h, py + search_radius)

        # Crop the frame. (NumPy slicing returns a view, so drawing
        # on roi_frame will also draw on the original frame.)
        roi_frame = frame[y_min:y_max, x_min:x_max]
        offset_x, offset_y = x_min, y_min
    else:
        roi_frame = frame
        offset_x, offset_y = 0, 0

    blurred = cv2.GaussianBlur(roi_frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Internal helper to find a specific color blob
    def get_color_center(lower, upper, lower2=None, upper2=None):
        mask = cv2.inRange(hsv, lower, upper)
        # Red requires two masks because its hue wraps around 0 and 180
        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            # Ignore tiny specks of noise (must be bigger than 5 pixels)
            if cv2.contourArea(largest) > 5:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None

    found_points = []

    # 1. Look for GREEN
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    green_c = get_color_center(lower_green, upper_green)
    if green_c:
        found_points.append(green_c)
        cv2.circle(roi_frame, green_c, 5, (0, 255, 0), -1)

    # 2. Look for BLUE / WHITE-BLUE
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    blue_c = get_color_center(lower_blue, upper_blue)
    if blue_c:
        found_points.append(blue_c)
        cv2.circle(roi_frame, blue_c, 5, (255, 0, 0), -1)

    # 3. Look for RED (hue wraps around 0 and 180 in HSV)
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 100, 100]), np.array([180, 255, 255])
    red_c = get_color_center(lower_red1, upper_red1, lower_red2, upper_red2)
    if red_c:
        found_points.append(red_c)
        cv2.circle(roi_frame, red_c, 5, (0, 0, 255), -1)

    # --- CALCULATE DRONE CENTER ---
    if len(found_points) > 0:
        avg_x = sum(p[0] for p in found_points) // len(found_points)
        avg_y = sum(p[1] for p in found_points) // len(found_points)

        # Convert local ROI coordinates back to full-frame coordinates
        return avg_x + offset_x, avg_y + offset_y

    # Nothing detected
    return None, None


# ==========================================
# CAMERA MANAGEMENT
# ==========================================
def open_cameras(front_index=1, side_index=2):
    """Open front and side USB cameras, returning (cam_front, cam_side)."""
    cam_front = cv2.VideoCapture(front_index)
    cam_side = cv2.VideoCapture(side_index)

    if not cam_front.isOpened():
        raise RuntimeError(f"Front camera (index {front_index}) failed to open")
    if not cam_side.isOpened():
        cam_front.release()
        raise RuntimeError(f"Side camera (index {side_index}) failed to open")

    return cam_front, cam_side


def release_cameras(*cams):
    """Release camera handles and close OpenCV windows."""
    for cam in cams:
        cam.release()
    cv2.destroyAllWindows()
