"""
Gaze Calibration Utility
=========================
Optional 5-point calibration for improved gaze accuracy.
Run this before the main gaze test for best results.

Usage:
    python utils/calibrate_gaze.py
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CALIBRATION_POINTS = [
    (0.1, 0.1),   # Top-left
    (0.9, 0.1),   # Top-right
    (0.5, 0.5),   # Center
    (0.1, 0.9),   # Bottom-left
    (0.9, 0.9),   # Bottom-right
]

WINDOW_W = 800
WINDOW_H = 600
COLLECT_FRAMES = 30   # Frames per calibration point
POINT_DISPLAY_SECS = 2.0


def run_calibration():
    """Run 5-point gaze calibration and save calibration data."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

    calibration_data = {}
    window_name = "ASD Gaze Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_W, WINDOW_H)

    print("Starting 5-point gaze calibration...")
    print("Look at each red dot when it appears.")

    for i, (nx, ny) in enumerate(CALIBRATION_POINTS):
        px, py = int(nx * WINDOW_W), int(ny * WINDOW_H)
        print(f"\nPoint {i+1}/5 at ({nx:.1f}, {ny:.1f}) - Look here!")

        collected_irises = []
        start = time.time()
        phase = "show"  # show -> collect -> done

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))
            display = frame.copy()

            elapsed = time.time() - start

            # Draw calibration point
            if phase == "show":
                cv2.circle(display, (px, py), 30, (0, 0, 200), 2)
                cv2.circle(display, (px, py), 15, (0, 0, 255), -1)
                cv2.putText(display, f"LOOK HERE - Point {i+1}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if elapsed > POINT_DISPLAY_SECS:
                    phase = "collect"
                    start = time.time()

            elif phase == "collect":
                cv2.circle(display, (px, py), 20, (0, 255, 0), -1)
                cv2.putText(display, f"Hold... {int(COLLECT_FRAMES - len(collected_irises))} frames",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Collect iris data
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]

                    # Left iris center
                    left_iris_pts = [(lm[idx].x * w, lm[idx].y * h) for idx in [474, 475, 476, 477]]
                    left_cx = np.mean([p[0] for p in left_iris_pts])
                    left_cy = np.mean([p[1] for p in left_iris_pts])

                    collected_irises.append((left_cx, left_cy))

                    if len(collected_irises) >= COLLECT_FRAMES:
                        phase = "done"

            elif phase == "done":
                cv2.putText(display, f"✓ Point {i+1} done!",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(window_name, display)
                cv2.waitKey(800)
                break

            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None

        if collected_irises:
            mean_iris = np.mean(collected_irises, axis=0).tolist()
            calibration_data[f"point_{i}"] = {
                "screen_pos": [nx, ny],
                "iris_pos": mean_iris,
                "samples": len(collected_irises)
            }
            print(f"  Collected {len(collected_irises)} samples, mean iris: {mean_iris}")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    # Save calibration
    os.makedirs("./data", exist_ok=True)
    cal_path = "./data/gaze_calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"\nCalibration complete! Saved to {cal_path}")
    print(f"Collected data for {len(calibration_data)} points.")
    return calibration_data


if __name__ == "__main__":
    run_calibration()
