"""
train.py

Builds (or rebuilds) the LBPH face recognizer model from every enrolled
profile's saved images. Run this any time you add/remove profiles or
photos, or just call train_model() from another script (enroll.py does
this automatically).

Usage:
    python3.14 train.py
"""

import os

import cv2
import numpy as np

from profile_manager import load_profiles, BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "trainer.yml")


def train_model():
    data = load_profiles()
    people = data["people"]

    if not people:
        print("No profiles found. Run enroll.py first.")
        return False

    faces = []
    labels = []

    for pid, info in people.items():
        photo_dir = info["photo_dir"]
        if not os.path.isdir(photo_dir):
            continue
        for fname in os.listdir(photo_dir):
            if not fname.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(photo_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(int(pid))

    if not faces:
        print("No training images found in any profile folder.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    print(f"Trained on {len(faces)} images across {len(people)} people.")
    print(f"Model saved to {MODEL_PATH}")
    return True


if __name__ == "__main__":
    train_model()
