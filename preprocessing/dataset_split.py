import os
import json
import shutil

INPUT_VIDEOS_DIR = "data/train_sample_videos"
RAW_DIR = "data/raw"
REAL_DIR = os.path.join(RAW_DIR, "real")
FAKE_DIR = os.path.join(RAW_DIR, "fake")

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

# Load metadata
with open(os.path.join(INPUT_VIDEOS_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

for filename, info in metadata.items():
    label = info["label"]
    src_path = os.path.join(INPUT_VIDEOS_DIR, filename)

    if not os.path.exists(src_path):
        continue

    if label == "REAL":
        shutil.move(src_path, os.path.join(REAL_DIR, filename))
    elif label == "FAKE":
        shutil.move(src_path, os.path.join(FAKE_DIR, filename))

print("✅ Videos separated into REAL and FAKE folders.")
