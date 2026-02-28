import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, num_frames=30):
    """Extracts a specific number of frames from a video at regular intervals."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return

    interval = max(1, total_frames // num_frames)
    count = 0
    saved_count = 0

    video_name = os.path.basename(video_path).split('.')[0]

    while saved_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % interval == 0:
            frame_name = f"{video_name}_frame_{saved_count}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1
        
        count += 1

    cap.release()

def process_dataset(raw_dir, processed_dir, num_frames=30):
    """Processes 'real' and 'fake' subdirectories in the raw_dir."""
    for label in ['real', 'fake']:
        input_subdir = os.path.join(raw_dir, label)
        output_subdir = os.path.join(processed_dir, label)

        if not os.path.exists(input_subdir):
            print(f"⚠️ Directory not found: {input_subdir}")
            continue

        print(f"📂 Processing {label} videos...")
        video_files = [f for f in os.listdir(input_subdir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_file in tqdm(video_files):
            video_path = os.path.join(input_subdir, video_file)
            extract_frames(video_path, output_subdir, num_frames)

if __name__ == "__main__":
    # Ensure we are in the deepfake-detection-system directory or paths are relative to it
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(base_dir, "data/raw")
    PROCESSED_DATA_DIR = os.path.join(base_dir, "data/processed")
    
    process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    print("🎉 Frame extraction complete!")