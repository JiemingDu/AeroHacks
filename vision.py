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

    # TUNE THESE TO MATCH YOUR EXACT GREEN LED IN THE CAGE!
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
# Connect to the two USB cameras
cam_front = cv2.VideoCapture(0)
cam_side = cv2.VideoCapture(1)

# Get camera resolution (usually 640x480)
FRAME_WIDTH = int(cam_front.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cam_front.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Target Pixels: The exact center of the video feeds
TARGET_ROLL_X = FRAME_WIDTH // 2  # e.g., 320
TARGET_PITCH_Y = FRAME_WIDTH // 2  # e.g., 320 (Using side camera's horizontal axis)
TARGET_ALTITUDE = FRAME_HEIGHT // 2  # e.g., 240 (Using front camera's vertical axis)

# Initialize your PIDs (Using the class from the previous answer)
# NOTE: Kp, Ki, Kd values are guesses. Start small!
pid_altitude = PIDController(Kp=0.05, Ki=0.0, Kd=0.01)
pid_roll = PIDController(Kp=0.02, Ki=0.0, Kd=0.01)
pid_pitch = PIDController(Kp=0.02, Ki=0.0, Kd=0.01)

# Flight variables
base_throttle = 150  # Starting guess for hover throttle
last_time = time.time()

# Tell drone to turn on the green LED, start balancing, and spool up motors
# green_LED(1)
# set_mode(2)
# manual_thrusts(base_throttle, base_throttle, base_throttle, base_throttle)

print(f"Targets set to: Center X/Y: {TARGET_ROLL_X}, Height Z: {TARGET_ALTITUDE}")

# ==========================================
# 3. MAIN FLIGHT LOOP
# ==========================================
while True:
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    # Grab images from both cameras
    ret_front, frame_front = cam_front.read()
    ret_side, frame_side = cam_side.read()

    if not ret_front or not ret_side:
        print("Camera glitch, skipping frame...")
        continue

    # Get the (x, y) pixel coordinates of the drone from both cameras
    front_x, front_y = get_drone_pixel_position(frame_front)
    side_x, side_y = get_drone_pixel_position(frame_side)

    # Only run control logic if BOTH cameras can see the drone
    if front_x is not None and side_x is not None:
        # --- CALCULATE ERRORS ---
        # OpenCV coords: (0,0) is TOP LEFT.
        # Moving Right increases X. Moving Down increases Y.

        error_roll_x = TARGET_ROLL_X - front_x
        error_pitch_y = TARGET_PITCH_Y - side_x
        error_altitude = TARGET_ALTITUDE - front_y

        # --- RUN PIDs ---
        roll_adj = pid_roll.compute(TARGET_ROLL_X, front_x, dt)
        pitch_adj = pid_pitch.compute(TARGET_PITCH_Y, side_x, dt)
        throttle_adj = pid_altitude.compute(TARGET_ALTITUDE, front_y, dt)

        # --- SEND COMMANDS TO DRONE ---

        # 1. Position Control (Pitch and Roll)
        # IMPORTANT: You must test if positive/negative commands push the drone the right way!
        # set_roll(roll_adj)
        # set_pitch(pitch_adj)

        # 2. Altitude Control
        # If target = 240, and drone drops to 300 (lower on screen). Error = 240 - 300 = -60.
        # PID output will be negative. But we need to ADD throttle to go up.
        # Therefore, we SUBTRACT the adjustment from the base throttle.
        new_throttle = base_throttle - throttle_adj

        # Clamp throttle to safe limits (0 to 250 max)
        new_throttle = max(0, min(250, new_throttle))

        # manual_thrusts(new_throttle, new_throttle, new_throttle, new_throttle)

    # --- VISUAL DEBUGGING (Shows on your laptop screen) ---
    if front_x is not None:
        cv2.circle(frame_front, (front_x, front_y), 10, (0, 0, 255), -1)
        cv2.putText(frame_front, f"Altitude Err: {error_altitude}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(frame_front, f"Roll Err: {error_roll_x}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if side_x is not None:
        cv2.circle(frame_side, (side_x, side_y), 10, (255, 0, 0), -1)
        cv2.putText(frame_side, f"Pitch Err: {error_pitch_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Front Camera (Roll & Altitude)", frame_front)
    cv2.imshow("Side Camera (Pitch)", frame_side)

    # Press 'ESC' to emergency stop and quit!
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        # emergency_stop()
        print("EMERGENCY STOP TRIGGERED")
        break

# Cleanup
cam_front.release()
cam_side.release()
cv2.destroyAllWindows()