import time
import drone

print("=== Drone Sanity Check ===\n")

# Firmware
version = drone.get_firmware_version()
print(f"Firmware version : {version}")

# Sensors at rest
pitch = drone.get_pitch()
roll  = drone.get_roll()
gyro_pitch = drone.get_gyro_pitch()
gyro_roll  = drone.get_gyro_roll()
print(f"Pitch            : {pitch:.3f}")
print(f"Roll             : {roll:.3f}")
print(f"Gyro pitch rate  : {gyro_pitch:.3f} deg/s")
print(f"Gyro roll rate   : {gyro_roll:.3f} deg/s")

# Mode switching
drone.set_mode(1)
time.sleep(0.2)
mode = drone.get_mode()
print(f"Mode after set_mode(1): {mode}  (expected 1)")
drone.set_mode(0)

# LEDs
print("\nFlashing LEDs...")
drone.red_LED(1);   time.sleep(0.3); drone.red_LED(0)
drone.green_LED(1); time.sleep(0.3); drone.green_LED(0)
drone.blue_LED(1);  time.sleep(0.3); drone.blue_LED(0)

# PID integral
drone.reset_integral()
i_vals = drone.get_i_values()
print(f"\nIntegrals after reset : {i_vals}  (expected [0.0, 0.0])")

# --- MOTOR ARM TEST: uncomment to spin motors briefly ---
# SAFE THROTTLE is low (50/250) — still keep props off or drone secured
# drone.set_mode(1)
# drone.manual_thrusts(50, 50, 50, 50)
# time.sleep(1)
# drone.emergency_stop()
# print("Motor arm test done.")
# ---------------------------------------------------------

print("\n=== Done. Drone left in mode 0 (off). ===")