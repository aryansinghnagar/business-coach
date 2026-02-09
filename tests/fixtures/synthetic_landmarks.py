"""
Synthetic landmark generator for metric validation tests.

Creates MediaPipe-style 468x3 face landmarks that produce known high/low
configurations for each signifier. Used to test ExpressionSignifierEngine
without real video.

Coordinate system: pixel space for 640x480 frame (engine normalizes if needed).
Indices follow MediaPipe face mesh (see utils/expression_signifiers.py).
"""

import numpy as np
import math
from typing import Tuple

# Key indices from expression_signifiers
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
LEFT_EYEBROW = [107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276]
NOSE = [4, 6, 19, 20, 51, 94, 168, 197, 326, 327, 358, 359, 360, 361]
MOUTH_LEFT, MOUTH_RIGHT = 61, 17
NOSE_TIP, CHIN = 4, 175
FACE_LEFT, FACE_RIGHT = 234, 454


def _make_base_landmarks(w: int = 640, h: int = 480) -> np.ndarray:
    """Create 468x3 landmark array with neutral face geometry in pixel space."""
    lm = np.zeros((468, 3), dtype=np.float64)
    cx, cy = w / 2, h / 2
    face_w = min(w, h) * 0.35
    face_h = face_w * 1.2

    # Left eye center (270), right eye center (370)
    lex, ley = cx - face_w * 0.25, cy - face_h * 0.1
    rex, rey = cx + face_w * 0.25, cy - face_h * 0.1
    eye_h = face_w * 0.08
    eye_w = face_w * 0.15

    # Open eyes: vertical span > horizontal for normal EAR
    for i, idx in enumerate(LEFT_EYE):
        t = i / max(len(LEFT_EYE) - 1, 1)
        lm[idx, 0] = lex - eye_w / 2 + eye_w * (t if t <= 0.5 else 1 - t)
        lm[idx, 1] = ley - eye_h / 2 + eye_h * (0.3 + 0.7 * (t if t < 0.5 else 1 - t))
        lm[idx, 2] = 0
    for i, idx in enumerate(RIGHT_EYE):
        t = i / max(len(RIGHT_EYE) - 1, 1)
        lm[idx, 0] = rex - eye_w / 2 + eye_w * (t if t <= 0.5 else 1 - t)
        lm[idx, 1] = rey - eye_h / 2 + eye_h * (0.3 + 0.7 * (t if t < 0.5 else 1 - t))
        lm[idx, 2] = 0

    # Eyebrows above eyes (deterministic)
    for i, idx in enumerate(LEFT_EYEBROW):
        lm[idx, 0] = lex - eye_w / 2 + (i / max(len(LEFT_EYEBROW) - 1, 1)) * eye_w * 0.5
        lm[idx, 1] = ley - eye_h - face_w * 0.03
        lm[idx, 2] = 0
    for i, idx in enumerate(RIGHT_EYEBROW):
        lm[idx, 0] = rex - eye_w / 2 + (i / max(len(RIGHT_EYEBROW) - 1, 1)) * eye_w * 0.5
        lm[idx, 1] = rey - eye_h - face_w * 0.03
        lm[idx, 2] = 0

    # Mouth (neutral)
    mw = face_w * 0.4
    mh = face_w * 0.08
    mx, my = cx, cy + face_h * 0.25
    for i, idx in enumerate(MOUTH):
        t = i / max(len(MOUTH) - 1, 1)
        lm[idx, 0] = mx - mw / 2 + mw * (t if t <= 0.5 else 1 - (t - 0.5) * 2)
        lm[idx, 1] = my - mh / 2 + mh * 0.5
        lm[idx, 2] = 0
    lm[MOUTH_LEFT, 0] = mx - mw / 2
    lm[MOUTH_LEFT, 1] = my
    lm[MOUTH_RIGHT, 0] = mx + mw / 2
    lm[MOUTH_RIGHT, 1] = my

    # Nose, chin
    lm[NOSE_TIP, 0] = cx
    lm[NOSE_TIP, 1] = cy + face_h * 0.05
    lm[NOSE_TIP, 2] = 0
    for idx in NOSE:
        if lm[idx, 0] == 0 and lm[idx, 1] == 0:
            lm[idx, 0] = cx + (np.random.rand() - 0.5) * face_w * 0.1
            lm[idx, 1] = cy + face_h * 0.05 + (np.random.rand() - 0.5) * face_h * 0.1
            lm[idx, 2] = 0
    lm[CHIN, 0] = cx
    lm[CHIN, 1] = cy + face_h * 0.5
    lm[CHIN, 2] = 0

    # Face outline
    lm[FACE_LEFT, 0] = cx - face_w / 2
    lm[FACE_LEFT, 1] = cy
    lm[FACE_LEFT, 2] = 0
    lm[FACE_RIGHT, 0] = cx + face_w / 2
    lm[FACE_RIGHT, 1] = cy
    lm[FACE_RIGHT, 2] = 0

    # Fill remaining with centroid (deterministic)
    np.random.seed(42)
    for i in range(lm.shape[0]):
        if np.all(lm[i] == 0):
            lm[i, 0] = cx + (np.random.rand() - 0.5) * face_w * 0.3
            lm[i, 1] = cy + (np.random.rand() - 0.5) * face_h * 0.3
            lm[i, 2] = 0
    return lm


def _rotate_landmarks_2d(lm: np.ndarray, cx: float, cy: float, roll_deg: float) -> np.ndarray:
    """Rotate landmarks around (cx,cy) by roll_deg."""
    out = lm.copy()
    rad = math.radians(roll_deg)
    c, s = math.cos(rad), math.sin(rad)
    for i in range(lm.shape[0]):
        x, y = lm[i, 0] - cx, lm[i, 1] - cy
        out[i, 0] = cx + x * c - y * s
        out[i, 1] = cy + x * s + y * c
    return out


def make_neutral_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    """Return 468x3 landmarks for neutral face."""
    h, w = shape[0], shape[1]
    return _make_base_landmarks(w, h)


def make_smile_landmarks(shape: Tuple[int, int, int] = (480, 640, 3), strength: float = 1.0) -> np.ndarray:
    """Return landmarks for Duchenne-like smile (raised lip corners, open mouth)."""
    lm = _make_base_landmarks(shape[1], shape[0])
    cx = shape[1] / 2
    # Raise mouth corners, increase mouth height (MAR)
    lm[MOUTH_LEFT, 1] -= 8 * strength
    lm[MOUTH_RIGHT, 1] -= 8 * strength
    for idx in MOUTH:
        lm[idx, 1] -= 5 * strength
    return lm


def make_head_tilt_landmarks(shape: Tuple[int, int, int] = (480, 640, 3), roll_deg: float = 12.0) -> np.ndarray:
    """Return landmarks for head tilted (lateral roll). Optimal engagement ~11° (Davidenko)."""
    h, w = shape[0], shape[1]
    lm = _make_base_landmarks(w, h)
    return _rotate_landmarks_2d(lm, w / 2, h / 2, roll_deg)


def make_gaze_aversion_landmarks(shape: Tuple[int, int, int] = (480, 640, 3), yaw_offset: float = 80) -> np.ndarray:
    """Return landmarks for gaze averted - shift nose/eyes relative to face outline (creates yaw)."""
    h, w = shape[0], shape[1]
    lm = _make_base_landmarks(w, h)
    # Shift inner face (eyes, nose, mouth, eyebrows) but NOT face outline
    inner_indices = set(LEFT_EYE + RIGHT_EYE + MOUTH + LEFT_EYEBROW + RIGHT_EYEBROW + NOSE + [NOSE_TIP, CHIN])
    for i in inner_indices:
        if i < lm.shape[0]:
            lm[i, 0] += yaw_offset
    # Keep FACE_LEFT, FACE_RIGHT fixed so (nose - face_cx) increases -> head_yaw
    return lm


def make_lip_compression_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    """Return landmarks for compressed lips (low MAR - narrow, flat mouth)."""
    lm = _make_base_landmarks(shape[1], shape[0])
    cx = shape[1] / 2
    my = np.mean([lm[i, 1] for i in MOUTH])
    mw_orig = lm[MOUTH_RIGHT, 0] - lm[MOUTH_LEFT, 0]
    # Narrow mouth significantly, collapse height (compressed)
    for idx in MOUTH:
        lm[idx, 0] = cx + (lm[idx, 0] - cx) * 0.4
        lm[idx, 1] = my + (lm[idx, 1] - my) * 0.2
    lm[MOUTH_LEFT, 0] = cx - mw_orig * 0.15
    lm[MOUTH_RIGHT, 0] = cx + mw_orig * 0.15
    return lm


def make_parted_lips_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    """Return landmarks for parted lips (high MAR - mouth height/width)."""
    lm = _make_base_landmarks(shape[1], shape[0])
    # Increase mouth height significantly to raise MAR (parted = open mouth)
    my = np.mean([lm[i, 1] for i in MOUTH])
    mw = lm[MOUTH_RIGHT, 0] - lm[MOUTH_LEFT, 0]
    for idx in MOUTH:
        if lm[idx, 1] > my:
            lm[idx, 1] += 25
        else:
            lm[idx, 1] -= 25
    # Widen mouth slightly for natural parted look
    cx = shape[1] / 2
    for idx in MOUTH:
        lm[idx, 0] = cx + (lm[idx, 0] - cx) * 1.15
    return lm


def make_contempt_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    """Return landmarks for unilateral lip raise (contempt - asymmetry)."""
    lm = _make_base_landmarks(shape[1], shape[0])
    # Raise one mouth corner only (Ekman unilateral) - strong asymmetry
    lm[MOUTH_RIGHT, 1] -= 25
    lm[MOUTH_RIGHT, 0] += 5
    # Lower opposite corner slightly to accentuate
    lm[MOUTH_LEFT, 1] += 8
    return lm


def make_brow_furrow_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    """Return landmarks for furrowed brow (lowered eyebrows)."""
    lm = _make_base_landmarks(shape[1], shape[0])
    for idx in LEFT_EYEBROW + RIGHT_EYEBROW:
        lm[idx, 1] += 10
    return lm


def make_brow_raise_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    """Return landmarks for raised eyebrows."""
    lm = _make_base_landmarks(shape[1], shape[0])
    for idx in LEFT_EYEBROW + RIGHT_EYEBROW:
        lm[idx, 1] -= 12
    return lm


def warmup_engine(engine, landmarks: np.ndarray, shape: Tuple[int, int, int], n_frames: int = 15) -> None:
    """Feed landmarks to engine for n_frames to fill buffer and establish baselines.
    Requires engine to have update(landmarks, face_result, frame_shape) method.
    """
    for _ in range(n_frames):
        engine.update(landmarks, None, shape)
