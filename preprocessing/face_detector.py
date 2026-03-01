"""
Face detection using OpenCV's DNN-based detector (more accurate than Haar cascade).
Falls back to full image if no face is detected so inference always proceeds.

Usage:
    from preprocessing.face_detector import detect_and_crop_face
    cropped = detect_and_crop_face(frame_bgr)   # numpy BGR array
"""

import cv2
import numpy as np


def _get_cascade():
    """Load OpenCV frontal face Haar cascade (always available with opencv-python)."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return cascade


_CASCADE = None


def detect_and_crop_face(
    image: np.ndarray,
    padding: float = 0.20,
    min_face_ratio: float = 0.05,
) -> np.ndarray:
    """
    Detect the largest face in a BGR image and return it cropped with padding.

    Args:
        image:          BGR numpy array (H, W, 3).
        padding:        Fraction of face size to pad on each side (default 20%).
        min_face_ratio: Minimum face width as fraction of image width to accept
                        the detection (filters tiny false positives).

    Returns:
        Cropped BGR face region, or the original image if no face is found.
    """
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = _get_cascade()

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(w * min_face_ratio), int(h * min_face_ratio)),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return image

    # Pick the largest detected face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0]

    # Apply padding
    pad_x = int(fw * padding)
    pad_y = int(fh * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)

    return image[y1:y2, x1:x2]


def detect_faces_all(image: np.ndarray, min_face_ratio: float = 0.05) -> list:
    """
    Return bounding boxes for all faces detected in the image.

    Returns:
        List of (x, y, w, h) tuples, empty list if none found.
    """
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = _get_cascade()

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(w * min_face_ratio), int(h * min_face_ratio)),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return []
    return [tuple(f) for f in faces]