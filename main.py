# pylint: disable=no-member
import cv2
import time
import drone
import controller
from vision import get_drone_pixel_position, open_cameras, release_cameras

BASE_THROTTLE  = 150
SEARCH_RADIUS  = 100   # pixels — ROI box half-size around last known position


def main():
    # --- Camera setup (USB cameras at index 1 and 2) ---
    try:
        cam_front, cam_side = open_cameras(front_index=1, side_index=2)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    FRONT_WIDTH  = int(cam_front.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRONT_HEIGHT = int(cam_front.get(cv2.CAP_PROP_FRAME_HEIGHT))
    SIDE_WIDTH   = int(cam_side.get(cv2.CAP_PROP_FRAME_WIDTH))
    SIDE_HEIGHT  = int(cam_side.get(cv2.CAP_PROP_FRAME_HEIGHT))

    TARGET_ROLL_X   = FRONT_WIDTH  // 2
    TARGET_PITCH_X  = SIDE_WIDTH   // 2
    TARGET_ALTITUDE = FRONT_HEIGHT // 2

    print(f"Front frame  : {FRONT_WIDTH} x {FRONT_HEIGHT}")
    print(f"Side  frame  : {SIDE_WIDTH} x {SIDE_HEIGHT}")
    print(f"Target center: Roll_X={TARGET_ROLL_X}, Pitch_X={TARGET_PITCH_X}, Alt={TARGET_ALTITUDE}")

    # --- Drone startup ---
    drone.green_LED(1)
    drone.set_mode(2)
    drone.manual_thrusts(BASE_THROTTLE, BASE_THROTTLE, BASE_THROTTLE, BASE_THROTTLE)

    # ROI memory — None means search the full frame
    prev_front_pos = None
    prev_side_pos  = None

    lost_frames = 0
    MAX_LOST    = 10

    # initialise error variables so debug overlay never crashes on first frame
    error_roll_x   = 0
    error_pitch_x  = 0
    error_altitude = 0

    try:
        while True:
            ret_front, frame_front = cam_front.read()
            ret_side,  frame_side  = cam_side.read()

            if not ret_front or not ret_side:
                print("Camera glitch, skipping frame...")
                continue

            # --- LED detection with ROI search ---
            front_x, front_y = get_drone_pixel_position(frame_front, prev_front_pos, SEARCH_RADIUS)
            side_x,  side_y  = get_drone_pixel_position(frame_side,  prev_side_pos,  SEARCH_RADIUS)

            # --- Watchdog ---
            if front_x is None or side_x is None:
                lost_frames += 1
                # reset ROI memory so next detection searches the full frame
                prev_front_pos = None
                prev_side_pos  = None
                if lost_frames >= MAX_LOST:
                    print("Lost drone for too many frames — EMERGENCY STOP")
                    drone.emergency_stop()
                    break
            else:
                lost_frames = 0

                # update ROI memory for next frame
                prev_front_pos = (front_x, front_y)
                prev_side_pos  = (side_x,  side_y)

                # compute errors for display
                error_roll_x   = TARGET_ROLL_X   - front_x
                error_pitch_x  = TARGET_PITCH_X  - side_x
                error_altitude = TARGET_ALTITUDE  - front_y

                # send control commands (dt handled internally by PID)
                controller.run_control(
                    front_x, front_y, side_x,
                    TARGET_ROLL_X, TARGET_PITCH_X, TARGET_ALTITUDE
                )

            # --- Debug overlay: front camera ---
            cv2.drawMarker(frame_front, (TARGET_ROLL_X, TARGET_ALTITUDE),
                           (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            if prev_front_pos is not None:
                cv2.circle(frame_front, (front_x, front_y), 10, (0, 0, 255), -1)
                pt1 = (max(0, front_x - SEARCH_RADIUS), max(0, front_y - SEARCH_RADIUS))
                pt2 = (min(FRONT_WIDTH,  front_x + SEARCH_RADIUS),
                       min(FRONT_HEIGHT, front_y + SEARCH_RADIUS))
                cv2.rectangle(frame_front, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame_front, f"Altitude Err: {error_altitude}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_front, f"Roll Err: {error_roll_x}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Debug overlay: side camera ---
            cv2.drawMarker(frame_side, (TARGET_PITCH_X, TARGET_ALTITUDE),
                           (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            if prev_side_pos is not None:
                cv2.circle(frame_side, (side_x, side_y), 10, (255, 0, 0), -1)
                pt1 = (max(0, side_x - SEARCH_RADIUS), max(0, side_y - SEARCH_RADIUS))
                pt2 = (min(SIDE_WIDTH,  side_x + SEARCH_RADIUS),
                       min(SIDE_HEIGHT, side_y + SEARCH_RADIUS))
                cv2.rectangle(frame_side, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame_side, f"Pitch Err: {error_pitch_x}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Front Camera (Roll & Altitude)", frame_front)
            cv2.imshow("Side Camera (Pitch)", frame_side)

            if cv2.waitKey(1) & 0xFF == 27:
                print("ESC pressed — EMERGENCY STOP")
                drone.emergency_stop()
                break

    except Exception as e:
        print(f"Unexpected error: {e}")
        drone.emergency_stop()

    finally:
        release_cameras(cam_front, cam_side)


if __name__ == "__main__":
    main()