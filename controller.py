import time
import drone
from pid import PIDController as PID

# ==========================================
# FLIGHT CONTROLLER
#
# The drone runs in mode 2: its onboard PID stabilises pitch and
# roll to the target angles we send via set_pitch() / set_roll().
#
# Our job (outer loop):
#   - Vision pixel error  → desired tilt angle  (P + I terms)
#   - Gyro angular rate   → damping             (D term)
#   - Altitude pixel error → throttle            (full vision PID)
#
# NOTE: each call to drone.get_gyro_*() is a WiFi round-trip.
#       Two extra reads per frame is acceptable; avoid adding more.
# ==========================================

BASE_THROTTLE = 150
MAX_THROTTLE  = 250
MIN_THROTTLE  = 0

# Converts pixel error to a target angle for the drone's onboard PID.
# TUNE: start very small (0.01–0.05) to avoid aggressive tilts.
PIXEL_TO_ANGLE = 0.03

# Safety clamp on the angle command sent to the drone (drone angle units).
MAX_ANGLE = 15.0

# Outer-loop PIDs for roll and pitch.
# P/I act on pixel error (converted to angle); D uses gyro rate.
pid_roll  = PID(Kp=1.0, Ki=0.01, Kd=0.3)
pid_pitch = PID(Kp=1.0, Ki=0.01, Kd=0.3)

# Altitude PID — vision only (no altimeter on board).
pid_altitude = PID(Kp=0.05, Ki=0.0, Kd=0.01)

last_control_time = None


def run_control(front_x, front_y, side_x,
                target_roll_x, target_pitch_x, target_altitude):
    """
    Computes and sends control commands to the drone.

    Pitch/Roll:  pixel error → angle PID (P/I from vision, D from gyro)
                 → set_pitch() / set_roll() (drone onboard PID tracks the angle)
    Altitude:    pixel error → our PID → manual_thrusts()

    Args:
        front_x, front_y : pixel coords from front camera
        side_x           : horizontal pixel from side camera
        target_roll_x    : desired front_x (center of front frame)
        target_pitch_x   : desired side_x  (center of side frame)
        target_altitude  : desired front_y (center height of front frame)
    """
    global last_control_time
    now = time.perf_counter()
    dt = 0.02 if last_control_time is None else min(now - last_control_time, 0.1)
    last_control_time = now

    # --- Roll (left/right from front camera) ---
    if front_x is not None:
        desired_roll = (target_roll_x - front_x) * PIXEL_TO_ANGLE
        roll_cmd = pid_roll.compute(desired_roll, 0.0, dt)
        roll_cmd = max(-MAX_ANGLE, min(MAX_ANGLE, roll_cmd))
        drone.set_roll(roll_cmd)
    else:
        drone.set_roll(0.0)
        pid_roll.reset()

    # --- Pitch (front/back from side camera) ---
    if side_x is not None:
        desired_pitch = (target_pitch_x - side_x) * PIXEL_TO_ANGLE
        pitch_cmd = pid_pitch.compute(desired_pitch, 0.0, dt)
        pitch_cmd = max(-MAX_ANGLE, min(MAX_ANGLE, pitch_cmd))
        drone.set_pitch(pitch_cmd)
    else:
        drone.set_pitch(0.0)
        pid_pitch.reset()

    # --- Altitude (vision-only PID, no altimeter) ---
    if front_y is not None:
        throttle_adj = pid_altitude.compute(target_altitude, front_y, dt)
        new_throttle = int(max(MIN_THROTTLE,
                               min(MAX_THROTTLE, BASE_THROTTLE - throttle_adj)))
    else:
        new_throttle = BASE_THROTTLE

    drone.manual_thrusts(new_throttle, new_throttle, new_throttle, new_throttle)


def reset_pids():
    """Resets all PID state. Call on re-arm or mode switch."""
    pid_roll.reset()
    pid_pitch.reset()
    pid_altitude.reset()
    drone.reset_integral()