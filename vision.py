# pylint: disable=no-member
import cv2
import time
import numpy as np


# ==========================================
# 1. HELPER FUNCTIONS
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

        # Crop the frame. (OpenCV trick: drawing on this cropped frame
        # will automatically draw on the main frame too!)
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
        cv2.circle(roi_frame, green_c, 5, (0, 255, 0), -1)  # Draw tiny green dot

    # 2. Look for BLUE / WHITE-BLUE
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    blue_c = get_color_center(lower_blue, upper_blue)
    if blue_c:
        found_points.append(blue_c)
        cv2.circle(roi_frame, blue_c, 5, (255, 0, 0), -1)  # Draw tiny blue dot

    # 3. Look for RED
    # Red is tricky in HSV. It lives at 0-10 AND 170-180.
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 100, 100]), np.array([180, 255, 255])
    red_c = get_color_center(lower_red1, upper_red1, lower_red2, upper_red2)
    if red_c:
        found_points.append(red_c)
        cv2.circle(roi_frame, red_c, 5, (0, 0, 255), -1)  # Draw tiny red dot

    # --- CALCULATE DRONE CENTER ---
    # If we found at least one LED, average their positions!
    if len(found_points) > 0:
        avg_x = sum([p[0] for p in found_points]) // len(found_points)
        avg_y = sum([p[1] for p in found_points]) // len(found_points)

        # Convert local ROI coordinates back to full-screen coordinates
        return avg_x + offset_x, avg_y + offset_y

    # If we saw absolutely nothing, return None
    return None, None


# ==========================================
# 2. SETUP & CALIBRATION
# ==========================================
cam_front = cv2.VideoCapture(0)
cam_side = cv2.VideoCapture(1)
time.sleep(2)

print("Front opened:", cam_front.isOpened())
print("Side opened:", cam_side.isOpened())

if not cam_front.isOpened() or not cam_side.isOpened():
    print("ERROR: Camera failed to open. Check permissions.")
    cam_front.release()
    cam_side.release()
    exit()

FRAME_WIDTH = int(cam_front.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cam_front.get(cv2.CAP_PROP_FRAME_HEIGHT))

TARGET_ROLL_X = FRAME_WIDTH // 2
TARGET_PITCH_Y = FRAME_WIDTH // 2
TARGET_ALTITUDE = FRAME_HEIGHT // 2

base_throttle = 150
last_time = time.time()

prev_front_pos = None
prev_side_pos = None

# Variables for the dynamic search area based on motor speed
# prev_motor_speed = 150
# current_motor_speed = 150

print(f"Targets set to: Center X/Y: {TARGET_ROLL_X}, Height Z: {TARGET_ALTITUDE}")

# ==========================================
# 3. MAIN FLIGHT LOOP
# ==========================================
while True:
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    ret_front, frame_front = cam_front.read()
    ret_side, frame_side = cam_side.read()

    if not ret_front or not ret_side:
        print("Camera glitch, skipping frame...")
        continue

    # --- DYNAMIC SEARCH AREA LOGIC (COMMENTED OUT) ---
    # delta_speed = abs(current_motor_speed - prev_motor_speed)
    # dynamic_search_radius = 100 + int(delta_speed * 2.0)
    # prev_motor_speed = current_motor_speed

    current_search_radius = 100

    front_x, front_y = get_drone_pixel_position(frame_front, prev_front_pos, current_search_radius)
    side_x, side_y = get_drone_pixel_position(frame_side, prev_side_pos, current_search_radius)

    if front_x is not None and side_x is not None:
        prev_front_pos = (front_x, front_y)
        prev_side_pos = (side_x, side_y)

        error_roll_x = TARGET_ROLL_X - front_x
        error_pitch_y = TARGET_PITCH_Y - side_x
        error_altitude = TARGET_ALTITUDE - front_y

        # roll_adj = pid_roll.compute(TARGET_ROLL_X, front_x, dt)
        # pitch_adj = pid_pitch.compute(TARGET_PITCH_Y, side_x, dt)
        # throttle_adj = pid_altitude.compute(TARGET_ALTITUDE, front_y, dt)

        # new_throttle = base_throttle - throttle_adj
        # new_throttle = max(0, min(250, new_throttle))

        # current_motor_speed = new_throttle

        # manual_thrusts(new_throttle, new_throttle, new_throttle, new_throttle)
    else:
        prev_front_pos = None
        prev_side_pos = None

    # --- VISUAL DEBUGGING ---
    if prev_front_pos is not None:
        cv2.drawMarker(frame_front, (TARGET_ROLL_X, TARGET_ALTITUDE), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # Draw a big white circle representing the "Averaged" Drone Center
        cv2.circle(frame_front, (front_x, front_y), 10, (255, 255, 255), 2)

        pt1 = (max(0, front_x - current_search_radius), max(0, front_y - current_search_radius))
        pt2 = (min(FRAME_WIDTH, front_x + current_search_radius), min(FRAME_HEIGHT, front_y + current_search_radius))
        cv2.rectangle(frame_front, pt1, pt2, (0, 255, 0), 2)

        cv2.putText(frame_front, f"Altitude Err: {error_altitude}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(frame_front, f"Roll Err: {error_roll_x}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if prev_side_pos is not None:
        cv2.drawMarker(frame_side, (TARGET_PITCH_Y, TARGET_ALTITUDE), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(frame_side, (side_x, side_y), 10, (255, 255, 255), 2)

        pt1 = (max(0, side_x - current_search_radius), max(0, side_y - current_search_radius))
        pt2 = (min(FRAME_WIDTH, side_x + current_search_radius), min(FRAME_HEIGHT, side_y + current_search_radius))
        cv2.rectangle(frame_side, pt1, pt2, (0, 255, 0), 2)

        cv2.putText(frame_side, f"Pitch Err: {error_pitch_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Front Camera", frame_front)
    cv2.imshow("Side Camera", frame_side)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("EMERGENCY STOP TRIGGERED")
        break

# Cleanup
cam_front.release()
cam_side.release()
cv2.destroyAllWindows()