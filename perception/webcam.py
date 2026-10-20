import cv2
import csv
import time
import numpy as np

from pathlib import Path

from perception.landmarks import (
    create_hand_tracker,
    extract_landmarks,
    normalize_landmarks
)


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "hand_landmarks.csv"

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    tracker = create_hand_tracker()
    start_time = time.monotonic()
    previous_timestamp = -1
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "timestamp_ms",
            "wrist_x",
            "wrist_y"
        ]
        for i in range(21):
            header += [
                f"landmark_{i}_x",
                f"landmark_{i}_y",
                f"landmark_{i}_z"
            ]
        writer.writerow(header)
        try:
            while True:
                success, frame = camera.read()

                if not success:
                    break

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )
                timestamp_ms = max(
                    timestamp_ms,
                    previous_timestamp + 1
                )
                previous_timestamp = timestamp_ms

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                landmarks = extract_landmarks(
                    tracker,
                    rgb,
                    timestamp_ms
                )

                if landmarks is not None:
                    normalized = normalize_landmarks(
                        landmarks
                    )

                    wrist_x = landmarks[0, 0]
                    wrist_y = landmarks[0, 1]

                    row = [
                        timestamp_ms,
                        wrist_x,
                        wrist_y,
                        *normalized.flatten()
                    ]

                    writer.writerow(row)
                    height, width = frame.shape[:2]

                    for point in landmarks:
                        x = int(point[0] * width)
                        y = int(point[1] * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )
                cv2.imshow(
                    "CS 4701 Hand Tracking",
                    frame
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            tracker.close()
            cv2.destroyAllWindows()

    print(f"Saved landmarks to {output_file}")

if __name__ == "__main__":
    main()
