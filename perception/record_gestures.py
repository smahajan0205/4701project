import cv2
import csv
import time
import argparse
import numpy as np
from pathlib import Path
from perception.landmarks import (
    create_hand_tracker,
    extract_landmarks,
    normalize_landmarks,
)

GESTURES = ["left", "right", "up", "down", "idle"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=1.5)
    args = parser.parse_args()

    output_dir = Path("data") / "pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    tracker = create_hand_tracker()

    print("Gesture Recording Tool")
    print("Press SPACE to record a gesture.")
    print("Press Q to quit.")

    session_start = time.monotonic()
    previous_timestamp = -1

    try:
        for gesture in GESTURES:
            for rep in range(args.reps):

                print(f"Recording {gesture}, " f"repetition {rep + 1}/{args.reps}")
                # wait for user to press SPACE
                while True:
                    success, frame = camera.read()

                    if not success:
                        raise RuntimeError("Camera read failed")

                    frame = cv2.flip(frame, 1)

                    cv2.putText(
                        frame,
                        f"{gesture.upper()} - Rep {rep + 1}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        "Press SPACE to record | Q to quit",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imshow("Gesture Recorder", frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        return

                    if key == 32:
                        break

                filename = f"{args.user}_{gesture}_{rep + 1:03d}.csv"

                filepath = output_dir / filename
                with open(filepath, "w", newline="") as file:
                    writer = csv.writer(file)
                    header = [
                        "timestamp_ms",
                        "wrist_x",
                        "wrist_y",
                        "hand_scale",
                        "hand_detected",
                    ]

                    for i in range(21):
                        header += [
                            f"landmark_{i}_x",
                            f"landmark_{i}_y",
                            f"landmark_{i}_z",
                        ]

                    writer.writerow(header)

                    recording_start = time.monotonic()

                    while time.monotonic() - recording_start < args.duration:
                        success, frame = camera.read()

                        if not success:
                            raise RuntimeError("Camera read failed")

                        frame = cv2.flip(frame, 1)

                        timestamp_ms = int((time.monotonic() - session_start) * 1000)
                        timestamp_ms = max(timestamp_ms, previous_timestamp + 1)
                        previous_timestamp = timestamp_ms

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        landmarks = extract_landmarks(tracker, rgb, timestamp_ms)

                        if landmarks is not None:
                            normalized = normalize_landmarks(landmarks)

                            wrist_x = landmarks[0, 0]
                            wrist_y = landmarks[0, 1]

                            hand_scale = np.linalg.norm(
                                landmarks[9, :2] - landmarks[0, :2]
                            )

                            row = [
                                timestamp_ms,
                                wrist_x,
                                wrist_y,
                                hand_scale,
                                1,
                                *normalized.flatten(),
                            ]

                            writer.writerow(row)

                        else:
                            # preserve frames with missing detections
                            writer.writerow([timestamp_ms, "", "", "", 0] + [""] * 63)

                        cv2.putText(
                            frame,
                            f"RECORDING: {gesture.upper()}",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                        )

                        cv2.imshow("Gesture Recorder", frame)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return

                print(f"Saved: {filepath}")

    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
