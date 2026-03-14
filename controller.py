import time
import drone
from pid import PIDController as PID

# ==========================================
# FLIGHT CONTROLLER
#
# The drone has its own onboard PID for pitch and roll (mode 2).
# We use set_pitch() and set_roll() to give it target angles,
# which we derive from camera pixel error.
#
# Altitude has no onboard PID, so we run our own here.
# ==========================================

BASE_THROTTLE = 150
MAX_THROTTLE  = 250
MIN_THROTTLE  = 0

# Converts pixel error to a target angle for the drone's onboard PID.
# TUNE: start very small (0.01–0.05) to avoid aggressive tilts.
PIXEL_TO_ANGLE = 0.03

pid_altitude = PID(Kp=0.05, Ki=0.0, Kd=0.01)
last_control_time = None


def run_control(front_x, front_y, side_x,
                target_roll_x, target_pitch_y, target_altitude):
    """
    Computes and sends control commands to the drone.

    Pitch/Roll: pixel error → target angle → drone's onboard PID (mode 2)
    Altitude:   pixel error → our PID → manual_thrusts()

    Args:
        front_x, front_y : pixel coords from front camera
        side_x           : horizontal pixel from side camera
        target_roll_x    : desired front_x (center of front frame)
        target_pitch_y   : desired side_x  (center of side frame)
        target_altitude  : desired front_y (center height of front frame)
    """
    # --- Roll (left/right from front camera) ---
    if front_x is not None:
        error_roll = target_roll_x - front_x
        drone.set_roll(error_roll * PIXEL_TO_ANGLE)
    else:
        # reset to level if detection unavailable
        drone.set_roll(0.0)

    # --- Pitch (front/back from side camera) ---
    if side_x is not None:
        error_pitch = target_pitch_y - side_x
        drone.set_pitch(error_pitch * PIXEL_TO_ANGLE)
    else:
        # reset to level if detection unavailable
        drone.set_pitch(0.0)

    # --- Altitude (our PID, time tracked internally) ---
    # OpenCV y increases downward.
    # drone below target → front_y > target_altitude → error negative → less throttle subtracted → more thrust
    global last_control_time
    now = time.perf_counter()
    dt = 0.02 if last_control_time is None else (now - last_control_time)
    last_control_time = now

    # compute PID throttle adjustment on a valid reading; otherwise keep baseline
    if front_y is not None:
        throttle_adj = pid_altitude.compute(target_altitude, front_y, dt)
        new_throttle = int(max(MIN_THROTTLE, min(MAX_THROTTLE, BASE_THROTTLE - throttle_adj)))
    else:
        new_throttle = BASE_THROTTLE

    drone.manual_thrusts(new_throttle, new_throttle, new_throttle, new_throttle)


def reset_pids():
    """Resets altitude PID state. Call on re-arm or mode switch."""
    pid_altitude.reset()
    drone.reset_integral()