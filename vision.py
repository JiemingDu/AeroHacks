import cv2
import time
import numpy as np


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def get_drone_pixel_position(frame, prev_pos=None, search_radius=100):
    """
    Finds the green LED. If prev_pos (x,y) is given, it only searches inside
    a box of 'search_radius' around that point to prevent teleporting glitches.
    """
    h, w = frame.shape[:2]

    # If we know where the drone was last frame, crop the image to that area!
    if prev_pos is not None:
        px, py = prev_pos
        # Calculate box boundaries, ensuring we don't go off the edge of the screen
        x_min = max(0, px - search_radius)
        y_min = max(0, py - search_radius)
        x_max = min(w, px + search_radius)
        y_max = min(h, py + search_radius)

        # Crop the frame
        roi_frame = frame[y_min:y_max, x_min:x_max]
        offset_x, offset_y = x_min, y_min
    else:
        # If we lost the drone, search the whole screen
        roi_frame = frame
        offset_x, offset_y = 0, 0

    # Process only the cropped area (or full frame if lost)
    blurred = cv2.GaussianBlur(roi_frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx_roi = int(M["m10"] / M["m00"])
            cy_roi = int(M["m01"] / M["m00"])

            # Add the offset back to get the absolute pixel position on the full screen
            cx_absolute = cx_roi + offset_x
            cy_absolute = cy_roi + offset_y
            return cx_absolute, cy_absolute

    return None, None


# ==========================================
# 2. SETUP & CALIBRATION
# ==========================================
cam_front = cv2.VideoCapture(0)
cam_side = cv2.VideoCapture(0)  # FIXED: Changed to 1 so it doesn't open the same camera twice
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

# Memory variables to remember where the drone was last frame
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
    # If the speed changed a lot, the box gets bigger. Base size is 100 pixels.
    # dynamic_search_radius = 100 + int(delta_speed * 2.0)
    # prev_motor_speed = current_motor_speed

    # For now, we hardcode the search radius to 100 pixels
    current_search_radius = 100

    # Find the drone, passing in its last known position and the search radius
    front_x, front_y = get_drone_pixel_position(frame_front, prev_front_pos, current_search_radius)
    side_x, side_y = get_drone_pixel_position(frame_side, prev_side_pos, current_search_radius)

    if front_x is not None and side_x is not None:
        # Save positions for the next loop so the bounding box follows the drone
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

        # current_motor_speed = new_throttle # Update current speed for the dynamic box logic

        # manual_thrusts(new_throttle, new_throttle, new_throttle, new_throttle)
    else:
        # If we lost the drone, reset the memory so the next frame searches the ENTIRE screen to find it again
        prev_front_pos = None
        prev_side_pos = None

    # --- VISUAL DEBUGGING ---
    if prev_front_pos is not None:
        # Draw the target center line
        cv2.drawMarker(frame_front, (TARGET_ROLL_X, TARGET_ALTITUDE), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # Draw the Drone tracking circle
        cv2.circle(frame_front, (front_x, front_y), 10, (0, 0, 255), -1)

        # Draw the Search Area Bounding Box (Green Box)
        pt1 = (max(0, front_x - current_search_radius), max(0, front_y - current_search_radius))
        pt2 = (min(FRAME_WIDTH, front_x + current_search_radius), min(FRAME_HEIGHT, front_y + current_search_radius))
        cv2.rectangle(frame_front, pt1, pt2, (0, 255, 0), 2)

        cv2.putText(frame_front, f"Altitude Err: {error_altitude}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(frame_front, f"Roll Err: {error_roll_x}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if prev_side_pos is not None:
        cv2.drawMarker(frame_side, (TARGET_PITCH_Y, TARGET_ALTITUDE), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(frame_side, (side_x, side_y), 10, (255, 0, 0), -1)

        pt1 = (max(0, side_x - current_search_radius), max(0, side_y - current_search_radius))
        pt2 = (min(FRAME_WIDTH, side_x + current_search_radius), min(FRAME_HEIGHT, side_y + current_search_radius))
        cv2.rectangle(frame_side, pt1, pt2, (0, 255, 0), 2)

        cv2.putText(frame_side, f"Pitch Err: {error_pitch_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Front Camera (Roll & Altitude)", frame_front)
    cv2.imshow("Side Camera (Pitch)", frame_side)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("EMERGENCY STOP TRIGGERED")
        break

# Cleanup
cam_front.release()
cam_side.release()
cv2.destroyAllWindows()