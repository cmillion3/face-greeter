"""
enroll.py

Captures training face images for a new person using the webcam, then
kicks off training automatically.

Usage:
    python3.14 enroll.py "Alice" --greeting "Hey there, {name}! Great to see you."

If --greeting is omitted, a default "Hi {name}!" is used.
"""

import argparse
import os
import sys
import time

import cv2

from profile_manager import add_profile
import train

CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
_LOCAL_CASCADE = os.path.join(os.path.dirname(os.path.abspath(__file__)), CASCADE_FILENAME)
_BUNDLED_CASCADE = cv2.data.haarcascades + CASCADE_FILENAME
CASCADE_PATH = _LOCAL_CASCADE if os.path.exists(_LOCAL_CASCADE) else _BUNDLED_CASCADE

if not os.path.exists(CASCADE_PATH):
    print(
        "ERROR: Haar cascade file not found.\n"
        f"Checked: {_LOCAL_CASCADE}\n"
        f"Checked: {_BUNDLED_CASCADE}\n"
        "Download it with:\n"
        "  curl -o haarcascade_frontalface_default.xml "
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
    )
    sys.exit(1)
TARGET_SAMPLES = 25          # number of face images to capture
CAPTURE_INTERVAL_SEC = 0.3   # minimum gap between saved samples


def enroll(name, greeting, camera_index=0):
    person_id, photo_dir = add_profile(name, greeting)
    print(f"Enrolling '{name}' (id={person_id}). Saving images to: {photo_dir}")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: could not open camera. Check the camera_index / connection.")
        sys.exit(1)

    existing = len([f for f in os.listdir(photo_dir) if f.endswith(".jpg")])
    saved = 0
    last_save = 0
    print("Look at the camera. Press 'q' to stop early.")

    while saved < TARGET_SAMPLES:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Captured {saved}/{TARGET_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Enroll - press q to stop", frame)

        now = time.time()
        if len(faces) == 1 and (now - last_save) > CAPTURE_INTERVAL_SEC:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            out_path = os.path.join(photo_dir, f"{existing + saved:03d}.jpg")
            cv2.imwrite(out_path, face_img)
            saved += 1
            last_save = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images for '{name}'.")

    if saved == 0:
        print("No face images captured, skipping training.")
        return

    print("Training recognizer...")
    train.train_model()
    print("Done. Run greet.py to test recognition.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a new face profile.")
    parser.add_argument("name", help="Person's name")
    parser.add_argument("--greeting", default=None,
                         help='Greeting template, e.g. "Hey {name}, welcome back!"')
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    enroll(args.name, args.greeting, args.camera)
