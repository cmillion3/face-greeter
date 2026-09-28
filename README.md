# Face Greeter (Unitree G1 prototype)

Recognizes enrolled people via webcam and greets them by name. Built on
OpenCV only (Haar cascade for detection, LBPH for recognition) so it
installs cleanly on Python 3.14 without needing dlib/cmake.

## Setup

```bash
cd face_greeter
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if you're inside the venv — it's only
needed if installing outside one.)

## 1. Enroll a person

```bash
python3.14 enroll.py "Alice" --greeting "Hey there, {name}! Great to see you."
```

This opens your webcam, captures ~25 face images, saves them under
`profiles/Alice/`, and automatically (re)trains the recognition model.
Repeat for each person you want the robot to know. Run it again with an
existing name to add more training images for that person.

## 2. Check enrolled profiles

```bash
python3.14 profile_manager.py
```

## 3. Run live recognition + greeting

```bash
python3.14 greet.py
```

Press `q` in the video window to quit. Use `--no-speak` to disable audio
and just print greetings, or `--threshold` to tune match strictness
(lower number = stricter match, fewer false positives but more missed
matches).

## How matching works

LBPH compares a live face against each person's stored samples and
returns a distance score (lower = more similar). If the best match is
below the threshold, it counts as a recognized match; otherwise the
person is labeled "Unknown." A per-person cooldown (30s by default)
stops it from repeating the same greeting every frame.

## Extending profiles

Edit `profiles.json` directly, or use `profile_manager.add_profile()`,
to customize each person's greeting template. `{name}` gets substituted
automatically, e.g.:

```json
"greeting": "Welcome back, {name}! Hope your day is going well."
```

## Wiring into the actual G1

`greet.py` currently uses a standard webcam (`cv2.VideoCapture`) and
local text-to-speech (`pyttsx3`) as stand-ins. See the NOTES comment at
the bottom of that file for the two integration points: swapping in the
G1's camera stream and its speaker/audio SDK. The detection/recognition/
greeting logic in between doesn't need to change.

## Tuning tips

- **Lighting matters a lot for LBPH.** Enroll people under similar
  lighting to where the robot will actually operate.
- If you get false matches between different people, lower `--threshold`
  (try 60-70).
- If real matches are being missed, raise it slightly, or enroll more
  varied sample photos (different angles/expressions) via `enroll.py`.
- For a large number of people (dozens+), LBPH accuracy degrades faster
  than embedding-based methods (e.g. dlib/face_recognition, or ArcFace
  via `insightface`). Worth revisiting once dlib has stable 3.14 wheels,
  or if you're open to compiling it from source.
