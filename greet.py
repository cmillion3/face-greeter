"""
greet.py

Live loop: opens the camera, detects faces, matches them against the
trained LBPH model, and greets recognized people (console print + optional
local TTS). This is the piece you'd eventually wire into the G1's actual
camera feed and speaker output — see the NOTES section at the bottom.

Usage:
    python3.14 greet.py
    python3.14 greet.py --no-speak      # text-only, no audio
    python3.14 greet.py --threshold 70  # tune match strictness (lower = stricter)
"""

import argparse
import os
import time

import cv2

from profile_manager import get_greeting, get_profile, BASE_DIR

CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
_LOCAL_CASCADE = os.path.join(BASE_DIR, CASCADE_FILENAME)
_BUNDLED_CASCADE = cv2.data.haarcascades + CASCADE_FILENAME
CASCADE_PATH = _LOCAL_CASCADE if os.path.exists(_LOCAL_CASCADE) else _BUNDLED_CASCADE
MODEL_PATH = os.path.join(BASE_DIR, "trainer.yml")

# LBPH "confidence" is actually a distance: LOWER means a closer match.
# 100 is a common default cutoff; tighten (e.g. 60-70) to reduce false positives.
DEFAULT_THRESHOLD = 80
GREETING_COOLDOWN_SEC = 30  # don't re-greet the same person more than once per this window


def speak(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        # TTS is optional - if it's not installed/working, just skip audio.
        print(f"(TTS unavailable: {e})")


def run(camera_index=0, threshold=DEFAULT_THRESHOLD, speak_enabled=True):
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run enroll.py to add people first.")
        return

    if not os.path.exists(CASCADE_PATH):
        print(
            "ERROR: Haar cascade file not found.\n"
            "Download it with:\n"
            "  curl -o haarcascade_frontalface_default.xml "
            "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
        )
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: could not open camera.")
        return

    last_greeted = {}  # person_id -> timestamp of last greeting
    print("Watching for faces. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            person_id, confidence = recognizer.predict(face_img)

            if confidence <= threshold:
                profile = get_profile(person_id)
                label = profile["name"] if profile else "Unknown"
                color = (0, 255, 0)

                now = time.time()
                if now - last_greeted.get(person_id, 0) > GREETING_COOLDOWN_SEC:
                    greeting = get_greeting(person_id)
                    print(f"[MATCH] {greeting}  (confidence={confidence:.1f})")
                    if speak_enabled:
                        speak(greeting)
                    last_greeted[person_id] = now
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Greeter - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live face recognition + greeting loop.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help="LBPH match distance cutoff, lower = stricter (default 80)")
    parser.add_argument("--no-speak", action="store_true", help="Disable text-to-speech output")
    args = parser.parse_args()

    run(camera_index=args.camera, threshold=args.threshold, speak_enabled=not args.no_speak)

# -----------------------------------------------------------------------
# NOTES for integrating with the actual Unitree G1:
#
# 1. Camera feed: replace cv2.VideoCapture(camera_index) with however the
#    G1 SDK exposes its camera stream (e.g. subscribing to an image/ROS
#    topic and converting each message to a numpy BGR array before passing
#    it into the same detection/recognition pipeline below).
#
# 2. Audio output: replace speak() with a call into the G1's audio/speaker
#    SDK instead of pyttsx3, so greetings play through the robot itself.
#
# 3. Everything else (detection, LBPH matching, profile lookup, cooldown
#    logic) is hardware-agnostic and can stay as-is.
# -----------------------------------------------------------------------
