import os
import cv2
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

RAW_REAL = "data/raw/real"
RAW_FAKE = "data/raw/fake"

PROCESSED_REAL = "data/processed/real"
PROCESSED_FAKE = "data/processed/fake"

NUM_FRAMES_PER_VIDEO = 30
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

os.makedirs(PROCESSED_REAL, exist_ok=True)
os.makedirs(PROCESSED_FAKE, exist_ok=True)


def process_one_video(args):
    """Extract frames from a single video using seek (only decode frames we need)."""
    video_path, output_folder = args
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return

    num_to_extract = min(NUM_FRAMES_PER_VIDEO, total_frames)
    interval = max(1, total_frames // num_to_extract)
    indices = [min(i * interval, total_frames - 1) for i in range(num_to_extract)]

    video_id = os.path.basename(video_path).rsplit(".", 1)[0]

    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (224, 224))
            frame_name = f"{video_id}_{idx}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
    finally:
        cap.release()


def extract_frames(video_folder, output_folder, label, num_workers=None):
    """Extract frames from all videos in video_folder using multiprocessing and seek."""
    video_files = [
        f
        for f in os.listdir(video_folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]
    if not video_files:
        print(f"  No video files found in {video_folder}")
        return

    tasks = [
        (os.path.join(video_folder, name), output_folder)
        for name in video_files
    ]

    workers = num_workers or max(1, cpu_count() - 1)
    with Pool(processes=workers) as pool:
        list(
            tqdm(
                pool.imap(process_one_video, tasks),
                total=len(tasks),
                desc=f"  {label}",
                unit="video",
            )
        )


if __name__ == "__main__":
    print("Processing REAL videos...")
    extract_frames(RAW_REAL, PROCESSED_REAL, "REAL")

    print("Processing FAKE videos...")
    extract_frames(RAW_FAKE, PROCESSED_FAKE, "FAKE")

    print("✅ Frame extraction completed.")
