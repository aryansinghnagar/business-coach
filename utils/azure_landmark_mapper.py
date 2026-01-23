"""
Map Azure Face API 27 landmarks to MediaPipe 468 layout.

Expression signifiers and the feature extractor expect MediaPipe-style indices.
Azure returns 27 points in a fixed order; this expands them into a 468x3 array
so downstream logic works without OOB/zeros for key regions.

Azure order (from azure_face_api.extract_landmarks_from_face):
  0 pupilLeft, 1 pupilRight, 2 noseTip, 3 mouthLeft, 4 mouthRight,
  5 eyebrowLeftOuter, 6 eyebrowLeftInner, 7 eyeLeftOuter, 8 eyeLeftTop, 9 eyeLeftBottom, 10 eyeLeftInner,
  11 eyebrowRightInner, 12 eyebrowRightOuter, 13 eyeRightInner, 14 eyeRightTop, 15 eyeRightBottom, 16 eyeRightOuter,
  17 noseRootLeft, 18 noseRootRight, 19 noseLeftAlarTop, 20 noseRightAlarTop, 21 noseLeftAlarOutTip, 22 noseRightAlarOutTip,
  23 upperLipTop, 24 upperLipBottom, 25 underLipTop, 26 underLipBottom
"""

from typing import Optional, Tuple
import numpy as np

# MediaPipe indices used by expression_signifiers and feature_extractor
NOSE = [4, 6, 19, 20, 51, 94, 168, 197, 326, 327, 358, 359, 360, 361]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276]
MOUTH_LEFT, MOUTH_RIGHT = 61, 17
UPPER_LIP, LOWER_LIP = 13, 14
NOSE_TIP, CHIN = 4, 175
FACE_LEFT, FACE_RIGHT = 234, 454


def expand_azure_landmarks_to_mediapipe(
    landmarks: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int, ...],
) -> np.ndarray:
    """
    Expand Azure 27 landmarks to a 468x3 MediaPipe-compatible array.

    Fills the indices used by expression signifiers and feature extractor.
    bbox: (left, top, width, height) or None. Used for CHIN, FACE_LEFT, FACE_RIGHT.
    frame_shape: (H, W, ...). Used for sane fallbacks when bbox is missing.
    """
    out = np.zeros((468, 3), dtype=np.float64)
    h, w = int(frame_shape[0]), int(frame_shape[1])
    n = landmarks.shape[0]
    # ensure 2 or 3 cols
    lm = landmarks[:, :2] if landmarks.shape[1] >= 2 else landmarks
    if lm.shape[1] == 2:
        z = np.zeros((n, 1), dtype=lm.dtype)
        lm = np.hstack([lm, z])

    def get(azure_idx: int) -> Optional[np.ndarray]:
        if azure_idx < n:
            return np.array([float(lm[azure_idx, 0]), float(lm[azure_idx, 1]), 0.0], dtype=np.float64)
        return None

    def first(*candidates: Optional[np.ndarray]) -> np.ndarray:
        for c in candidates:
            if c is not None:
                return c
        return np.zeros(3, dtype=np.float64)

    def set_mp(mp_idx: int, pt: np.ndarray) -> None:
        if 0 <= mp_idx < 468:
            out[mp_idx, :] = pt[:3]

    # Nose: 2=noseTip, 17–22 nose roots/alar
    for i in NOSE:
        set_mp(i, first(get(2), get(17), get(18)))
    set_mp(4, first(get(2), get(17)))
    set_mp(6, first(get(17), get(2)))
    set_mp(19, first(get(18), get(2)))
    set_mp(20, first(get(19), get(2)))
    set_mp(51, first(get(21), get(2)))
    set_mp(94, first(get(22), get(2)))

    # Mouth: 3=mouthLeft, 4=mouthRight, 23–26 lips
    set_mp(MOUTH_LEFT, first(get(3)))
    set_mp(MOUTH_RIGHT, first(get(4)))
    set_mp(UPPER_LIP, first(get(23), get(24)))
    set_mp(LOWER_LIP, first(get(26), get(25)))
    fallback_mouth = first(get(24), get(3), get(4))
    for i in MOUTH:
        if out[i, 0] == 0 and out[i, 1] == 0:
            set_mp(i, fallback_mouth)

    # Left eye: 7–10
    pts = [get(7), get(8), get(9), get(10)]
    p0 = first(pts[0], pts[1], pts[2], pts[3])
    for i, idx in enumerate(LEFT_EYE):
        set_mp(idx, first(pts[i % 4], p0))

    # Right eye: 13–16
    pts = [get(16), get(14), get(15), get(13)]
    p0 = first(pts[0], pts[1], pts[2], pts[3])
    for i, idx in enumerate(RIGHT_EYE):
        set_mp(idx, first(pts[i % 4], p0))

    # Eyebrows: 5,6 and 11,12
    for i, idx in enumerate(LEFT_EYEBROW):
        set_mp(idx, first(get(6), get(5)) if i < 3 else first(get(5), get(6)))
    for i, idx in enumerate(RIGHT_EYEBROW):
        set_mp(idx, first(get(11), get(12)) if i < 3 else first(get(12), get(11)))

    # CHIN, FACE_LEFT, FACE_RIGHT from bbox or landmarks
    nose_pt = get(2)
    if n > 4:
        mouth_mid_x = (float(lm[3, 0]) + float(lm[4, 0])) / 2
    else:
        mouth_mid_x = (float(out[61, 0]) + float(out[17, 0])) / 2 if (out[61, 0] != 0 or out[17, 0] != 0) else w / 2
    if n > 26:
        mouth_bottom_y = float(lm[26, 1])
    elif n > 24:
        mouth_bottom_y = (float(lm[24, 1]) + float(lm[25, 1])) / 2
    else:
        mouth_bottom_y = float(out[14, 1]) if (out[14, 0] != 0 or out[14, 1] != 0) else h * 0.55
    nose_y = float(nose_pt[1]) if nose_pt is not None else (float(lm[24, 1]) if n > 24 else mouth_bottom_y - 20.0)

    if bbox is not None and len(bbox) >= 4:
        left, top, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        set_mp(CHIN, np.array([mouth_mid_x, mouth_bottom_y + 0.15 * max(bh, 1), 0.0], dtype=np.float64))
        set_mp(FACE_LEFT, np.array([float(left), nose_y, 0.0], dtype=np.float64))
        set_mp(FACE_RIGHT, np.array([float(left + bw), nose_y, 0.0], dtype=np.float64))
    else:
        set_mp(CHIN, np.array([mouth_mid_x, mouth_bottom_y + 0.1 * h, 0.0], dtype=np.float64))
        set_mp(FACE_LEFT, np.array([max(0, mouth_mid_x - 0.3 * w), nose_y, 0.0], dtype=np.float64))
        set_mp(FACE_RIGHT, np.array([min(w, mouth_mid_x + 0.3 * w), nose_y, 0.0], dtype=np.float64))

    return out
