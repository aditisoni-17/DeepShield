"""
Split processed/real and processed/fake images into train/val/test (70/15/15) by video.
Creates data/processed/{train,val,test}/{real,fake}/ so training/dataset.py can use ImageFolder.
Run after frame_extractor.py.
"""

import os
import shutil
import random

PROCESSED_REAL = "data/processed/real"
PROCESSED_FAKE = "data/processed/fake"
OUTPUT_ROOT = "data/processed"

SPLITS = ("train", "val", "test")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _video_id_from_filename(filename: str) -> str:
    """Extract video_id from frame name like 'video_abc_0.jpg' -> 'video_abc'."""
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else base


def _build_video_to_files(class_dir: str) -> dict[str, list[str]]:
    """List images in class_dir and map video_id -> list of image filenames."""
    video_to_files: dict[str, list[str]] = {}
    if not os.path.isdir(class_dir):
        return video_to_files
    for name in os.listdir(class_dir):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            vid = _video_id_from_filename(name)
            video_to_files.setdefault(vid, []).append(name)
    return video_to_files


def _split_video_ids(video_ids: list[str]) -> dict[str, list[str]]:
    """Split video_ids into train/val/test with 70/15/15. Returns {split: [video_ids]}."""
    random.seed(RANDOM_STATE)
    ids = list(video_ids)
    random.shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * TEST_RATIO))
    n_val = max(1, round(n * VAL_RATIO))
    n_train = n - n_test - n_val
    n_train = max(1, n_train)
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def _copy_split(
    class_dir: str,
    video_to_files: dict[str, list[str]],
    split_video_ids: dict[str, list[str]],
    class_name: str,
) -> None:
    """Create split dirs and copy images for this class."""
    for split in SPLITS:
        out_dir = os.path.join(OUTPUT_ROOT, split, class_name)
        os.makedirs(out_dir, exist_ok=True)
        for vid in split_video_ids.get(split, []):
            for fname in video_to_files.get(vid, []):
                src = os.path.join(class_dir, fname)
                dst = os.path.join(out_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)


def run_split(use_symlinks: bool = False) -> None:
    """Build train/val/test from processed/real and processed/fake by video (70/15/15)."""
    if use_symlinks:
        raise NotImplementedError("Symlinks not implemented; use copy.")

    for class_name, class_dir in [("real", PROCESSED_REAL), ("fake", PROCESSED_FAKE)]:
        if not os.path.isdir(class_dir):
            print(f"  Skipping {class_dir} (not a directory)")
            continue
        video_to_files = _build_video_to_files(class_dir)
        if not video_to_files:
            print(f"  No images found in {class_dir}")
            continue
        video_ids = list(video_to_files.keys())
        split_ids = _split_video_ids(video_ids)
        _copy_split(class_dir, video_to_files, split_ids, class_name)
        n_train = len(split_ids["train"])
        n_val = len(split_ids["val"])
        n_test = len(split_ids["test"])
        print(f"  {class_name}: {len(video_ids)} videos -> train={n_train}, val={n_val}, test={n_test}")

    print("✅ Train/val/test split created under data/processed/.")


if __name__ == "__main__":
    run_split()
