"""
profile_manager.py

Handles the JSON "database" of person profiles. Each profile has:
  - id: integer label used internally by the LBPH recognizer
  - name: person's name (used in the greeting)
  - greeting: custom greeting template, "{name}" gets substituted
  - photo_dir: folder containing that person's training face images

Run this file directly to list current profiles:
    python3.14 profile_manager.py
"""

import argparse
import json
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
PROFILES_JSON = os.path.join(BASE_DIR, "profiles.json")


def load_profiles():
    """Return the profiles dict, creating an empty one if it doesn't exist yet."""
    if not os.path.exists(PROFILES_JSON):
        return {"next_id": 0, "people": {}}
    with open(PROFILES_JSON, "r") as f:
        return json.load(f)


def save_profiles(data):
    with open(PROFILES_JSON, "w") as f:
        json.dump(data, f, indent=2)


def add_profile(name, greeting=None):
    """
    Create a new profile, or update+reuse the existing one if the name is
    already enrolled (a new --greeting on re-enroll will update it).
    Returns (profile_id, photo_dir).
    """
    data = load_profiles()

    # Reuse existing profile if this name is already enrolled
    for pid, info in data["people"].items():
        if info["name"].lower() == name.lower():
            if greeting:
                info["greeting"] = greeting
                save_profiles(data)
            return int(pid), info["photo_dir"]

    new_id = data["next_id"]
    photo_dir = os.path.join(PROFILES_DIR, name.replace(" ", "_"))
    os.makedirs(photo_dir, exist_ok=True)

    data["people"][str(new_id)] = {
        "name": name,
        "greeting": greeting or f"Hi {{name}}!",
        "photo_dir": photo_dir,
    }
    data["next_id"] = new_id + 1
    save_profiles(data)
    return new_id, photo_dir


def get_profile(person_id):
    data = load_profiles()
    return data["people"].get(str(person_id))


def get_greeting(person_id):
    profile = get_profile(person_id)
    if not profile:
        return None
    return profile["greeting"].format(name=profile["name"])


def delete_profile(name):
    """
    Delete a person's profile: removes their entry from profiles.json and
    deletes their saved training photos. Retrains the model afterward
    (or removes the model file entirely if no one is left).
    """
    data = load_profiles()
    target_id = None
    for pid, info in data["people"].items():
        if info["name"].lower() == name.lower():
            target_id = pid
            break

    if target_id is None:
        print(f"No profile found for '{name}'.")
        return False

    photo_dir = data["people"][target_id]["photo_dir"]
    if os.path.isdir(photo_dir):
        shutil.rmtree(photo_dir)

    del data["people"][target_id]
    save_profiles(data)
    print(f"Deleted profile '{name}' (id={target_id}) and its photos.")

    # Retrain on whoever's left, or clear the model if nobody remains.
    import train  # deferred import to avoid circular import at module load time
    if data["people"]:
        train.train_model()
    else:
        if os.path.exists(train.MODEL_PATH):
            os.remove(train.MODEL_PATH)
        print("No profiles left; removed trained model.")

    return True


def list_profiles():
    data = load_profiles()
    if not data["people"]:
        print("No profiles enrolled yet. Run enroll.py first.")
        return
    print(f"{'ID':<4} {'Name':<20} Greeting")
    print("-" * 50)
    for pid, info in sorted(data["people"].items(), key=lambda x: int(x[0])):
        print(f"{pid:<4} {info['name']:<20} {info['greeting']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View or manage face profiles.")
    parser.add_argument("--delete", metavar="NAME", help="Delete a person's profile and photos")
    args = parser.parse_args()

    if args.delete:
        delete_profile(args.delete)
    else:
        list_profiles()
