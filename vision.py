import cv2
import time
import numpy as np


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def get_drone_pixel_position(frame):
    """Finds the green LED and returns its (x, y) pixel coordinates."""
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None, None


# ==========================================
# 2. SETUP & CALIBRATION
# ==========================================
cam_front = cv2.VideoCapture(0)
cam_side = cv2.VideoCapture(0)
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

    front_x, front_y = get_drone_pixel_position(frame_front)
    side_x, side_y = get_drone_pixel_position(frame_side)

    if front_x is not None and side_x is not None:
        error_roll_x = TARGET_ROLL_X - front_x
        error_pitch_y = TARGET_PITCH_Y - side_x
        error_altitude = TARGET_ALTITUDE - front_y

        #roll_adj = pid_roll.compute(TARGET_ROLL_X, front_x, dt)
        #pitch_adj = pid_pitch.compute(TARGET_PITCH_Y, side_x, dt)
        #throttle_adj = pid_altitude.compute(TARGET_ALTITUDE, front_y, dt)

        #new_throttle = base_throttle - throttle_adj
        #new_throttle = max(0, min(250, new_throttle))
        #manual_thrusts(new_throttle, new_throttle, new_throttle, new_throttle)

    if front_x is not None:
        cv2.circle(frame_front, (front_x, front_y), 10, (0, 0, 255), -1)
        cv2.putText(frame_front, f"Altitude Err: {error_altitude}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_front, f"Roll Err: {error_roll_x}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if side_x is not None:
        cv2.circle(frame_side, (side_x, side_y), 10, (255, 0, 0), -1)
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