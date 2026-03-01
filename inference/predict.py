"""
Single-image and video inference: load model, same preprocessing as training, output Real/Fake + confidence.
Run from project root: python -m inference.predict path/to/image.jpg
"""

import os
import sys

import torch
from PIL import Image
from torchvision import transforms

# Allow running from project root or from inference/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.cnn_model import DeepfakeCNN

# Match training preprocessing (dataset.py)
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def load_model(model_path="saved_models/best_model.pth", device=None):
    """Load trained DeepfakeCNN from checkpoint."""
    if device is None:
        device = _get_device()
    model = DeepfakeCNN().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


def preprocess_image(image_input, transform=None):
    """
    image_input: path (str), PIL Image, or numpy array (BGR/RGB).
    Returns: tensor (1, 3, 224, 224) on CPU (call .to(device) before model).
    """
    if transform is None:
        transform = get_transform()

    if isinstance(image_input, str):
        pil = Image.open(image_input).convert("RGB")
    elif hasattr(image_input, "convert"):
        pil = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
    else:
        import numpy as np
        arr = image_input if isinstance(image_input, np.ndarray) else np.array(image_input)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        pil = Image.fromarray(arr)
    tensor = transform(pil).unsqueeze(0)
    return tensor


def predict(model, image_tensor, device):
    """
    Run model forward; image_tensor shape (1, 3, 224, 224).
    Returns: label ("Real" or "Fake"), confidence in [0, 1].
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logit = model(image_tensor)[0, 0].item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    label = "Real" if prob >= 0.5 else "Fake"
    confidence = prob if label == "Real" else (1.0 - prob)
    return label, confidence


def predict_image(model_path, image_path, device=None):
    """
    Load model, load image, preprocess, predict.
    Returns: dict with keys: label, confidence, prob_real.
    """
    model, device = load_model(model_path, device)
    transform = get_transform()
    tensor = preprocess_image(image_path, transform)
    label, confidence = predict(model, tensor, device)
    with torch.no_grad():
        logit = model(tensor.to(device))[0, 0].item()
    prob_real = torch.sigmoid(torch.tensor(logit)).item()
    return {"label": label, "confidence": confidence, "prob_real": prob_real}


def predict_with_gradcam(model_path, image_path, device=None):
    """
    Same as predict_image but also returns Grad-CAM heatmap and overlay.
    Returns: dict with label, confidence, prob_real, heatmap (H,W), overlay (numpy BGR).
    """
    from explainability.gradcam import generate_gradcam
    from explainability.heatmap_utils import overlay_heatmap
    import numpy as np
    import cv2

    model, device = load_model(model_path, device)
    transform = get_transform()
    tensor = preprocess_image(image_path, transform).to(device)
    tensor.requires_grad_(True)

    heatmap = generate_gradcam(model, tensor, target_layer=model.backbone.features[-1])
    with torch.no_grad():
        logit = model(tensor.detach())[0, 0].item()
    prob_real = torch.sigmoid(torch.tensor(logit)).item()
    label = "Real" if prob_real >= 0.5 else "Fake"
    confidence = prob_real if label == "Real" else (1.0 - prob_real)

    if isinstance(image_path, str):
        img_bgr = cv2.imread(image_path)
    else:
        img_bgr = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    if img_bgr is None:
        img_bgr = (tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(heatmap, img_bgr, alpha=0.5)

    return {
        "label": label,
        "confidence": confidence,
        "prob_real": prob_real,
        "heatmap": heatmap,
        "overlay": overlay,
    }


def predict_video(model_path, video_path, num_frames=16, device=None):
    """
    Sample num_frames evenly across the video, run inference on each, and
    return an aggregated result plus per-frame breakdown.

    Returns dict with keys:
        label        – majority-vote label ("Real" or "Fake")
        confidence   – average confidence across sampled frames
        prob_real    – average P(Real) across sampled frames
        frame_results – list of {frame_idx, label, confidence, prob_real}
        frames_analyzed – number of frames actually analyzed
    """
    import cv2

    model, device = load_model(model_path, device)
    transform = get_transform()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video has no readable frames.")

    # Sample frame indices evenly across the video
    sample_count = min(num_frames, total_frames)
    indices = [int(i * total_frames / sample_count) for i in range(sample_count)]

    frame_results = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess_image(frame_rgb, transform)
        with torch.no_grad():
            logit = model(tensor.to(device))[0, 0].item()
        prob_real = torch.sigmoid(torch.tensor(logit)).item()
        lbl = "Real" if prob_real >= 0.5 else "Fake"
        conf = prob_real if lbl == "Real" else (1.0 - prob_real)
        frame_results.append({
            "frame_idx": idx,
            "label": lbl,
            "confidence": round(conf, 4),
            "prob_real": round(prob_real, 4),
        })

    cap.release()

    if not frame_results:
        return {"label": "Unknown", "confidence": 0.0, "prob_real": 0.0,
                "frame_results": [], "frames_analyzed": 0}

    avg_prob_real = sum(r["prob_real"] for r in frame_results) / len(frame_results)
    avg_confidence = sum(r["confidence"] for r in frame_results) / len(frame_results)
    real_votes = sum(1 for r in frame_results if r["label"] == "Real")
    majority_label = "Real" if real_votes > len(frame_results) / 2 else "Fake"

    return {
        "label": majority_label,
        "confidence": round(avg_confidence, 4),
        "prob_real": round(avg_prob_real, 4),
        "frame_results": frame_results,
        "frames_analyzed": len(frame_results),
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not os.path.isfile(path):
        print("Usage: python -m inference.predict <image_path>")
        sys.exit(1)
    out = predict_image("saved_models/best_model.pth", path)
    print(f"Label: {out['label']}  Confidence: {out['confidence']:.3f}  P(Real): {out['prob_real']:.3f}")
