"""
Expression Signifiers Module

Implements 30 B2B-relevant facial expression signifiers from MediaPipe/Azure Face API
landmarks. Each signifier is scored 0-100. Psychology-informed weights and formulas
(see docs/DOCUMENTATION.md §7).

We keep a short history of recent frames (a "buffer") so we can detect patterns over
time—e.g. nodding, blinks, or sustained stillness. Weights control how much each
signifier contributes to the final engagement score (configurable per signifier and per group).

Groups:
  1. Interest & Engagement (10): eye contact (strong), head tilt, nodding, Duchenne
     (mouth-primary, squinch secondary), forward lean, parted lips, symmetry, etc.
  2. Cognitive Load (7): look away, thinking brow, eye squint, lip pucker, stillness.
  3. Resistance (9): gaze aversion (strong), lip compression, contempt, jaw clench, etc.
  4. Decision-Ready (3): smile transition, fixed gaze, relaxed exhale.
"""

import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from utils.face_detection_interface import FaceDetectionResult


# MediaPipe-style indices (use safe access for Azure's 27-point set)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
LEFT_EYEBROW = [107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276]
NOSE = [4, 6, 19, 20, 51, 94, 168, 197, 326, 327, 358, 359, 360, 361]
# Mouth corners: 61 (L), 291 (L inner), 17 (R), 78 (R inner) – 291,78 may be out of MOUTH
MOUTH_LEFT, MOUTH_RIGHT = 61, 17
NOSE_TIP, CHIN = 4, 175
FACE_LEFT, FACE_RIGHT = 234, 454
# Inner brows for furrow
INNER_BROW_L, INNER_BROW_R = 70, 300  # fallback to first/last of eyebrow if OOB

# All 30 signifier keys (single shared list; no per-call allocation).
SIGNIFIER_KEYS: List[str] = [
    "g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
    "g1_forward_lean", "g1_facial_symmetry", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead",
    "g2_look_up_lr", "g2_lip_pucker", "g2_eye_squint", "g2_thinking_brow", "g2_chin_stroke",
    "g2_stillness", "g2_lowered_brow",
    "g3_contempt", "g3_nose_crinkle", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
    "g3_rapid_blink", "g3_gaze_aversion", "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover",
    "g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition",
]


def _safe(landmarks: np.ndarray, indices: List[int], dim: int = 2) -> np.ndarray:
    n, d = landmarks.shape[0], min(dim, landmarks.shape[1])
    out = []
    for i in indices:
        if i < n:
            out.append(landmarks[i, :d])
        else:
            out.append(np.zeros(d, dtype=landmarks.dtype))
    return np.array(out) if out else np.zeros((0, d))


def _ear(pts: np.ndarray) -> float:
    """Eye aspect ratio; robust to degenerate/outlier points."""
    if len(pts) < 4:
        return 0.2
    x, y = pts[:, 0], pts[:, 1]
    v = np.abs(np.max(y) - np.min(y))
    h = np.max(x) - np.min(x) + 1e-6
    ear = v / max(h, 2.0)  # Floor h to avoid noise-driven extremes
    return float(np.clip(ear, 0.05, 0.8))  # Plausible range


def _median_recent(buf: list, key: str, n: int = 3, default: float = 0.0) -> float:
    """Median of last n buffer values for key; robust to outliers."""
    if not buf or n < 1:
        return default
    vals = [b.get(key, default) for b in buf[-n:]]
    return float(np.median(vals))


def _normalize_lm(landmarks: np.ndarray, w: int, h: int) -> np.ndarray:
    lm = np.array(landmarks, dtype=np.float64)
    if lm.shape[1] == 2:
        lm = np.hstack([lm, np.zeros((lm.shape[0], 1))])
    if np.max(lm[:, :2]) <= 1.0:
        lm[:, 0] *= w
        lm[:, 1] *= h
        if lm.shape[1] > 2:
            lm[:, 2] *= max(w, h)
    return lm


class ExpressionSignifierEngine:
    """
    Computes 30 expression signifier scores and a composite engagement score (0-100)
    from landmarks and optional FaceDetectionResult. Uses a temporal buffer (recent
    frames) for time-based signifiers like blinks, nodding, and stillness; weights
    determine how much each signifier affects the final score.
    """

    def __init__(
        self,
        buffer_frames: int = 22,
        weights_provider: Optional[Callable[[], Dict[str, List[float]]]] = None,
    ):
        self.buffer_frames = max(10, buffer_frames)
        self._buf: deque = deque(maxlen=self.buffer_frames)
        self._weights_provider = weights_provider
        self._landmarks: Optional[np.ndarray] = None
        self._face_result: Optional[FaceDetectionResult] = None
        self._shape: Optional[Tuple[int, int, int]] = None
        # Baseline for dilation proxy and Z
        self._baseline_eye_area: float = 0.0
        self._baseline_z: float = 0.0
        self._baseline_ear: float = 0.0  # For Duchenne squinch detection
        self._baseline_mar: float = 0.0  # For Duchenne mouth-opening comparison
        self._pupil_dilation_history: deque = deque(maxlen=3)  # Minimal smoothing for real-time
        self._last_pupil_dilation_score: float = 0.0  # Last valid score (for blink frames); default 0
        self._blink_start_frames: int = 0
        self._blinks_in_window: int = 0
        self._last_blink_reset: float = 0.0
        # Feature smoothing: blend current with previous (0.75 = 75% new, 25% prior)
        self._smooth_alpha: float = 0.75
        # Contempt: baseline asymmetry (person-specific); history for temporal consistency
        self._contempt_asymmetry_history: deque = deque(maxlen=24)
        # Hysteresis for binary 0/100: avoid flip when raw hovers near cutoff (up 58, down 48)
        self._binary_state: Dict[str, float] = {}
        # Previous-frame landmarks for temporal movement (relaxed exhale: stillness after release)
        self._prev_landmarks: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear temporal buffer and baselines (e.g. on detection start)."""
        self._buf.clear()
        self._baseline_eye_area = 0.0
        self._baseline_z = 0.0
        self._baseline_ear = 0.0
        self._baseline_mar = 0.0
        self._pupil_dilation_history.clear()
        self._last_pupil_dilation_score = 0.0  # default 0; output scale 0–100
        self._blink_start_frames = 0
        self._blinks_in_window = 0
        self._contempt_asymmetry_history.clear()
        self._binary_state.clear()
        self._prev_landmarks = None

    def update(
        self,
        landmarks: np.ndarray,
        face_result: Optional[FaceDetectionResult],
        frame_shape: Tuple[int, int, int],
    ) -> None:
        h, w = frame_shape[:2]
        lm = _normalize_lm(landmarks, w, h)
        self._landmarks = lm
        self._face_result = face_result
        self._shape = frame_shape

        # Snapshot for buffer
        le = _safe(lm, LEFT_EYE)
        re = _safe(lm, RIGHT_EYE)
        left_ear = _ear(le) if len(le) >= 4 else 0.2
        right_ear = _ear(re) if len(re) >= 4 else 0.2
        ear = (left_ear + right_ear) / 2.0
        eye_area = 0.0
        if len(le) >= 3 and len(re) >= 3:
            def _area(p):
                if len(p) < 3: return 0.0
                x, y = p[:, 0], p[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            eye_area = _area(le[:, :2]) + _area(re[:, :2])

        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        ley = np.mean(lb[:, 1]) if len(lb) else 0.0
        rey = np.mean(rb[:, 1]) if len(rb) else 0.0
        leye_y = np.mean(le[:, 1]) if len(le) else 0.0
        reye_y = np.mean(re[:, 1]) if len(re) else 0.0
        eyebrow_l = leye_y - ley if (ley != 0 or leye_y != 0) else 15.0
        eyebrow_r = reye_y - rey if (rey != 0 or reye_y != 0) else 15.0

        mouth_pts = _safe(lm, MOUTH)
        if len(mouth_pts) < 4:
            mar, mw, mh = 0.2, 40.0, 8.0
        else:
            mw = float(np.max(mouth_pts[:, 0]) - np.min(mouth_pts[:, 0]) + 1e-6)
            mh = float(np.max(mouth_pts[:, 1]) - np.min(mouth_pts[:, 1]))
            mar = float(np.clip(mh / mw, 0.02, 0.65))  # Plausible range; rejects noise extremes

        # Head pose from participation-style logic
        # Try to use Azure head_pose if available (more accurate)
        if face_result and face_result.head_pose:
            head_pitch = float(face_result.head_pose.get('pitch', 0.0))
            head_yaw = float(face_result.head_pose.get('yaw', 0.0))
            head_roll = float(abs(face_result.head_pose.get('roll', 0.0)))
        else:
            # MediaPipe: estimate from landmarks
            nose = lm[NOSE_TIP, :2] if lm.shape[0] > NOSE_TIP else lm[0, :2]
            chin = lm[CHIN, :2] if lm.shape[0] > CHIN else lm[min(16, lm.shape[0] - 1), :2]
            lf = lm[FACE_LEFT, 0] if lm.shape[0] > FACE_LEFT else np.mean(lm[:, 0]) - 50
            rf = lm[FACE_RIGHT, 0] if lm.shape[0] > FACE_RIGHT else np.mean(lm[:, 0]) + 50
            face_cx = (lf + rf) / 2
            face_hw = max(1e-6, abs(rf - lf) / 2)
            head_yaw = np.clip((nose[0] - face_cx) / face_hw, -1.5, 1.5) * 45.0
            
            # For pitch: use nose position relative to eye center (better than chin-nose)
            # Azure convention: positive pitch = looking up, negative = looking down
            # Looking up: nose moves up (lower Y) relative to eyes
            # Looking down: nose moves down (higher Y) relative to eyes
            eye_center_y = (np.mean(le[:, 1]) + np.mean(re[:, 1])) / 2 if (len(le) and len(re)) else nose[1]
            nose_offset_y = eye_center_y - nose[1]  # Positive = nose above eyes (looking up), Negative = nose below (looking down)
            # Normalize by face scale for pitch estimation
            face_scale_for_pitch = max(1e-6, face_hw * 2.0)  # Use face width as proxy for scale
            head_pitch = np.clip(nose_offset_y / face_scale_for_pitch * 45.0, -30.0, 30.0)  # Scale to degrees, match Azure convention
            
            head_roll = 0.0
            if lm.shape[0] > max(FACE_LEFT, FACE_RIGHT):
                ly = float(lm[FACE_LEFT, 1])
                ry = float(lm[FACE_RIGHT, 1])
                head_roll = abs(np.degrees(np.arctan2(ry - ly, rf - lf)))

        face_z = float(np.mean(lm[:, 2])) if lm.shape[1] > 2 and np.any(np.isfinite(lm[:, 2])) else 0.0
        frame_c = np.array([w / 2, h / 2])
        eye_c = (np.mean(le[:, :2], axis=0) + np.mean(re[:, :2], axis=0)) / 2 if (len(le) and len(re)) else frame_c
        gaze_x = float(eye_c[0] - frame_c[0])
        gaze_y = float(eye_c[1] - frame_c[1])
        face_var = float(np.var(lm[:, :2])) if lm.size >= 4 else 0.0
        nose_arr = [lm[i, 1] for i in NOSE if i < lm.shape[0]]
        nose_std = float(np.std(nose_arr)) if nose_arr else 0.0
        nose_height = float(max(nose_arr) - min(nose_arr)) if nose_arr else 0.0
        
        # Face scale factor for size/position invariance (inter-ocular distance)
        # This normalizes all pixel-based thresholds to be relative to face size
        if len(le) >= 2 and len(re) >= 2:
            left_eye_center = np.mean(le[:, :2], axis=0)
            right_eye_center = np.mean(re[:, :2], axis=0)
            face_scale = float(np.linalg.norm(left_eye_center - right_eye_center))
        elif mw > 0:
            face_scale = mw * 1.5  # Fallback: mouth width * 1.5 (typical ratio)
        else:
            # Last resort: estimate from landmark spread
            if lm.shape[0] > 10:
                face_scale = float(np.max(lm[:min(100, lm.shape[0]), 0]) - np.min(lm[:min(100, lm.shape[0]), 0])) * 0.4
            else:
                face_scale = 50.0
        face_scale = max(20.0, face_scale)  # Minimum scale to avoid division issues
        # Proper MAR: V=d(13,14) / H=d(61,17). Better for lip compression than bbox MAR.
        mar_inner = mar
        if lm.shape[0] > 61:
            p13 = lm[13, :2]
            p14 = lm[14, :2]
            p61 = lm[61, :2]
            p17 = lm[17, :2]
            v = float(np.linalg.norm(p13 - p14))
            h = float(np.linalg.norm(p61 - p17)) + 1e-6
            if h >= 5.0:
                mar_inner = v / h
        is_blink = 1.0 if ear < 0.16 else 0.0

        # Mouth corner asymmetry ratio (for contempt baseline): |ly - ry| / face_scale
        mouth_corner_asymmetry_ratio = 0.0
        if lm.shape[0] > max(MOUTH_LEFT, MOUTH_RIGHT):
            ly = float(lm[MOUTH_LEFT, 1])
            ry = float(lm[MOUTH_RIGHT, 1])
            mouth_corner_asymmetry_ratio = abs(ly - ry) / max(face_scale, 1e-6)

        # Baselines (EMA 0.92/0.08) - slower adaptation for noise invariance; sensitive to sustained changes
        if is_blink < 0.5:
            if self._baseline_eye_area <= 0 and eye_area > 0:
                self._baseline_eye_area = eye_area
            elif eye_area > 0:
                self._baseline_eye_area = 0.92 * self._baseline_eye_area + 0.08 * eye_area
        if self._baseline_z == 0 and face_z != 0:
            self._baseline_z = face_z
        elif face_z != 0:
            self._baseline_z = 0.92 * self._baseline_z + 0.08 * face_z
        if is_blink < 0.5:
            if self._baseline_ear <= 0 and ear > 0:
                self._baseline_ear = ear
            elif ear > 0:
                self._baseline_ear = 0.92 * self._baseline_ear + 0.08 * ear
        if self._baseline_mar <= 0 and mar_inner > 0:
            self._baseline_mar = mar_inner
        elif mar_inner > 0:
            self._baseline_mar = 0.92 * self._baseline_mar + 0.08 * mar_inner

        # Feature-level temporal smoothing: blend with previous frame to reduce landmark jitter
        prev = self._buf[-1] if len(self._buf) > 0 else None
        a = self._smooth_alpha
        if prev is not None:
            ear = a * ear + (1 - a) * prev.get("ear", ear)
            eye_area = a * eye_area + (1 - a) * prev.get("eye_area", eye_area)
            mar = a * mar + (1 - a) * prev.get("mar", mar)
            mar_inner = a * mar_inner + (1 - a) * prev.get("mar_inner", mar_inner)
            eyebrow_l = a * eyebrow_l + (1 - a) * prev.get("eyebrow_l", eyebrow_l)
            eyebrow_r = a * eyebrow_r + (1 - a) * prev.get("eyebrow_r", eyebrow_r)
            gaze_x = a * gaze_x + (1 - a) * prev.get("gaze_x", gaze_x)
            gaze_y = a * gaze_y + (1 - a) * prev.get("gaze_y", gaze_y)
            head_pitch = a * head_pitch + (1 - a) * prev.get("pitch", head_pitch)
            head_yaw = a * head_yaw + (1 - a) * prev.get("yaw", head_yaw)
            face_z = a * face_z + (1 - a) * prev.get("face_z", face_z)
            face_var = a * face_var + (1 - a) * prev.get("face_var", face_var)
            nose_std = a * nose_std + (1 - a) * prev.get("nose_std", nose_std)

        # Per-frame face movement (mean squared displacement vs previous frame) for relaxed exhale
        face_movement = 0.0
        if self._prev_landmarks is not None and self._prev_landmarks.shape[0] == lm.shape[0] and lm.shape[0] >= 4:
            n = min(lm.shape[0], self._prev_landmarks.shape[0])
            d = lm[:n, :2].astype(np.float64) - self._prev_landmarks[:n, :2].astype(np.float64)
            face_movement = float(np.mean(np.sum(d ** 2, axis=1)))

        t = time.time()
        # Blink counting (reset every 2s)
        if t - self._last_blink_reset > 2.0:
            self._blinks_in_window = 0
            self._last_blink_reset = t
        if is_blink > 0.5:
            self._blink_start_frames += 1
        else:
            if self._blink_start_frames >= 2 and self._blink_start_frames <= 8:
                self._blinks_in_window += 1
            self._blink_start_frames = 0

        snap = {
            "ear": ear, "eye_area": eye_area, "eyebrow_l": eyebrow_l, "eyebrow_r": eyebrow_r,
            "pitch": head_pitch, "yaw": head_yaw, "roll": head_roll,
            "mar": mar, "mar_inner": mar_inner, "mouth_w": mw, "mouth_h": mh,
            "face_z": face_z, "gaze_x": gaze_x, "gaze_y": gaze_y,
            "face_var": face_var, "nose_std": nose_std, "nose_height": nose_height, "is_blink": is_blink,
            "face_scale": face_scale,
            "mouth_corner_asymmetry_ratio": mouth_corner_asymmetry_ratio,
            "face_movement": face_movement,
        }
        self._buf.append(snap)
        self._prev_landmarks = lm.copy()

    def get_all_scores(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        lm = self._landmarks
        fr = self._face_result
        shp = self._shape
        buf = list(self._buf)
        if lm is None or shp is None or len(buf) < 1:
            for k in self._all_keys():
                out[k] = 0.0
            return out

        h, w = shp[0], shp[1]
        cur = buf[-1]
        mouth_pts = _safe(lm, MOUTH)
        ml = lm[MOUTH_LEFT, :2] if lm.shape[0] > MOUTH_LEFT else np.zeros(2)
        mr = lm[MOUTH_RIGHT, :2] if lm.shape[0] > MOUTH_RIGHT else np.zeros(2)

        # --- Group 1: Interest & Engagement ---
        out["g1_duchenne"] = self._g1_duchenne(lm, cur, buf)
        out["g1_pupil_dilation"] = self._g1_pupil_dilation(cur, buf)
        out["g1_eyebrow_flash"] = self._g1_eyebrow_flash(buf)
        out["g1_eye_contact"] = self._g1_eye_contact(cur, w, h, buf)
        out["g1_head_tilt"] = self._g1_head_tilt(cur, buf)
        out["g1_forward_lean"] = self._g1_forward_lean(cur, buf)
        out["g1_facial_symmetry"] = self._g1_facial_symmetry(lm, buf)
        out["g1_rhythmic_nodding"] = self._g1_rhythmic_nodding(buf)
        out["g1_parted_lips"] = self._g1_parted_lips(cur, buf)
        out["g1_softened_forehead"] = self._g1_softened_forehead(lm, cur, buf)

        # --- Group 2: Cognitive Load ---
        out["g2_look_up_lr"] = self._g2_look_up_lr(cur, buf)
        out["g2_lip_pucker"] = self._g2_lip_pucker(lm, cur, buf)
        out["g2_eye_squint"] = self._g2_eye_squint(cur, buf)
        out["g2_thinking_brow"] = self._g2_thinking_brow(lm, buf)
        out["g2_chin_stroke"] = 0.0  # no hand detection
        out["g2_stillness"] = self._g2_stillness(buf)
        out["g2_lowered_brow"] = self._g2_lowered_brow(lm, buf)

        # --- Group 3: Resistance (store as-is; composite uses 100 - x) ---
        out["g3_contempt"] = self._g3_contempt(lm, ml, mr, fr, cur, buf)
        out["g3_nose_crinkle"] = self._g3_nose_crinkle(cur, buf)
        out["g3_lip_compression"] = self._g3_lip_compression(cur, buf)
        out["g3_eye_block"] = self._g3_eye_block(buf)
        out["g3_jaw_clench"] = self._g3_jaw_clench(lm, cur, fr, buf)
        out["g3_rapid_blink"] = self._g3_rapid_blink()
        out["g3_gaze_aversion"] = self._g3_gaze_aversion(cur, buf)
        out["g3_no_nod"] = self._g3_no_nod(buf)
        out["g3_narrowed_pupils"] = self._g3_narrowed_pupils(cur, buf)  # proxy: squint
        out["g3_mouth_cover"] = 0.0  # no hand detection

        # --- Group 4: Decision-Ready ---
        out["g4_relaxed_exhale"] = self._g4_relaxed_exhale(buf)
        out["g4_fixed_gaze"] = self._g4_fixed_gaze(buf, w, h)
        out["g4_smile_transition"] = self._g4_smile_transition(buf, out.get("g1_duchenne", 0))

        # Binary output: 0 or 100 only; hysteresis to reduce flip when raw hovers near cutoff
        _UP, _DOWN = 58.0, 48.0  # raw: >= UP -> 100, <= DOWN -> 0, else keep previous (default 0)
        for k in out:
            v = float(out[k])
            if not np.isfinite(v):
                out[k] = 0.0
                self._binary_state[k] = 0.0
                continue
            prev = self._binary_state.get(k, 0.0)
            if v >= _UP:
                out[k] = 100.0
                self._binary_state[k] = 100.0
            elif v <= _DOWN:
                out[k] = 0.0
                self._binary_state[k] = 0.0
            else:
                out[k] = prev
        return out

    def _all_keys(self) -> List[str]:
        """Return the list of all 30 signifier keys (cached at module level)."""
        return SIGNIFIER_KEYS

    def _get_weights(self) -> Dict[str, List[float]]:
        """Fetch weights once per frame; reuse via get_group_means(..., W) / get_composite_score(..., W)."""
        return self._weights_provider() if self._weights_provider else {"signifier": [1.0] * 30, "group": [0.35, 0.15, 0.35, 0.15]}

    def get_group_means(self, scores: Optional[Dict[str, float]] = None, W: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
        """Return group means only (g1..g4). g3 is inverted (high = low resistance). No composite. Fast path for spike detection."""
        if scores is None:
            scores = self.get_all_scores()
        keys = self._all_keys()
        if W is None:
            W = self._get_weights()
        sw = W.get("signifier", [1.0] * 30)
        if len(sw) != 30:
            sw = [1.0] * 30

        def wmean(grp_keys: List[str]) -> float:
            total_w, total_ws = 0.0, 0.0
            for k in grp_keys:
                if k not in scores:
                    continue
                i = keys.index(k) if k in keys else 0
                wi = float(sw[i]) if i < len(sw) else 1.0
                total_w += wi * float(scores[k])
                total_ws += wi
            if total_ws > 1e-9:
                return total_w / total_ws
            vals = [float(scores[k]) for k in grp_keys if k in scores]
            return float(np.mean(vals)) if vals else 0.0

        g1k = ["g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
               "g1_forward_lean", "g1_facial_symmetry", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead"]
        g2k = ["g2_look_up_lr", "g2_lip_pucker", "g2_eye_squint", "g2_thinking_brow", "g2_chin_stroke",
               "g2_stillness", "g2_lowered_brow"]
        g3k = ["g3_contempt", "g3_nose_crinkle", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
               "g3_rapid_blink", "g3_gaze_aversion", "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover"]
        g4k = ["g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition"]

        g1 = wmean(g1k)
        g2 = wmean(g2k)
        g3_raw = wmean(g3k)
        g3 = 100.0 - g3_raw
        g4 = wmean(g4k)
        return {"g1": g1, "g2": g2, "g3": g3, "g4": g4}

    def _composite_from_group_means(self, g: Dict[str, float], gw: List[float]) -> float:
        """
        Composite engagement from group means. Psychology-informed adjustments:
        - High interest+decision-ready (G1+G4): bonus (convergent positive signals)
        - High resistance (G3_raw): penalty (contempt, gaze aversion, etc.)
        """
        composite_raw = gw[0] * g["g1"] + gw[1] * g["g2"] + gw[2] * g["g3"] + gw[3] * g["g4"]
        if (g["g1"] + g["g4"]) / 2.0 > 62.0:
            composite_raw = min(100.0, composite_raw + 8.0)
        if (100.0 - g["g3"]) > 38.0:
            composite_raw = max(0.0, composite_raw - 10.0)
        return float(max(0.0, min(100.0, composite_raw)))

    def get_composite_score(self, scores: Optional[Dict[str, float]] = None, W: Optional[Dict[str, List[float]]] = None) -> float:
        if scores is None:
            scores = self.get_all_scores()
        keys = self._all_keys()
        if all(float(scores.get(k, 0)) == 0.0 for k in keys):
            return 0.0
        if W is None:
            W = self._get_weights()
        gw = W.get("group", [0.35, 0.15, 0.35, 0.15])
        if len(gw) != 4:
            gw = [0.35, 0.15, 0.35, 0.15]
        g = self.get_group_means(scores, W)
        return self._composite_from_group_means(g, gw)

    def _get_composite_raw_for_breakdown(self, scores: Optional[Dict[str, float]], gw: List[float], g: Dict[str, float]) -> float:
        composite_raw = gw[0] * g["g1"] + gw[1] * g["g2"] + gw[2] * g["g3"] + gw[3] * g["g4"]
        if (g["g1"] + g["g4"]) / 2.0 > 62.0:
            composite_raw = min(100.0, composite_raw + 8.0)
        if (100.0 - g["g3"]) > 38.0:
            composite_raw = max(0.0, composite_raw - 10.0)
        return float(max(0.0, min(100.0, composite_raw)))

    def get_composite_breakdown(self, scores: Optional[Dict[str, float]] = None, W: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """
        Return a step-by-step breakdown of how the composite engagement score is calculated.
        Used for frontend "How is the score calculated?" and transparency.
        """
        if scores is None:
            scores = self.get_all_scores()
        keys = self._all_keys()
        if W is None:
            W = self._get_weights()
        sw = W.get("signifier", [1.0] * 30)
        gw = W.get("group", [0.35, 0.15, 0.35, 0.15])
        if len(sw) != 30:
            sw = [1.0] * 30
        if len(gw) != 4:
            gw = [0.35, 0.15, 0.35, 0.15]

        def weighted_mean(grp_keys: List[str]) -> float:
            total_w, total_ws = 0.0, 0.0
            for k in grp_keys:
                if k not in scores:
                    continue
                i = keys.index(k) if k in keys else 0
                wi = float(sw[i]) if i < len(sw) else 1.0
                si = float(scores[k])
                total_w += wi * si
                total_ws += wi
            if total_ws > 1e-9:
                return total_w / total_ws
            return float(np.mean([scores[k] for k in grp_keys if k in scores])) if grp_keys else 0.0

        g1k = ["g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
               "g1_forward_lean", "g1_facial_symmetry", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead"]
        g2k = ["g2_look_up_lr", "g2_lip_pucker", "g2_eye_squint", "g2_thinking_brow", "g2_chin_stroke",
               "g2_stillness", "g2_lowered_brow"]
        g3k = ["g3_contempt", "g3_nose_crinkle", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
               "g3_rapid_blink", "g3_gaze_aversion", "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover"]
        g4k = ["g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition"]

        g1 = weighted_mean(g1k)
        g2 = weighted_mean(g2k)
        g3_raw_wmean = weighted_mean(g3k)
        g3 = 100.0 - g3_raw_wmean
        g4 = weighted_mean(g4k)

        composite_before = gw[0] * g1 + gw[1] * g2 + gw[2] * g3 + gw[3] * g4
        adjustments: List[str] = []
        composite_after = composite_before
        if (g1 + g4) / 2.0 > 62.0:
            composite_after = min(100.0, composite_after + 8.0)
            adjustments.append("(G1+G4)/2 > 62: +8")
        if g3_raw_wmean > 38.0:
            composite_after = max(0.0, composite_after - 10.0)
            adjustments.append("G3_raw > 38: -10")
        score = float(max(0.0, min(100.0, composite_after)))

        return {
            "formula": "score = clip(G1*w1 + G2*w2 + G3*w3 + G4*w4 + adjustments, 0, 100)",
            "groupWeights": {"G1": gw[0], "G2": gw[1], "G3": gw[2], "G4": gw[3]},
            "groupMeans": {"G1": round(g1, 2), "G2": round(g2, 2), "G3_raw": round(g3_raw_wmean, 2), "G3": round(g3, 2), "G4": round(g4, 2)},
            "compositeBeforeAdjustments": round(composite_before, 2),
            "adjustments": adjustments if adjustments else ["none"],
            "score": round(score, 1),
            "signifierScores": {k: round(float(scores.get(k, 0)), 1) for k in keys if k in scores},
        }

    # ----- Group 1 -----
    def _g1_duchenne(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        """
        Duchenne smile detector based on FACS AU 6 (cheek raise/eye squinch) + AU 12 (lip corner pull).
        
        Output spans 0-100 after scaling:
        - Low (0-25): Neither corner lift nor squinch present
        - Medium (25-60): Only corner lift OR only squinch
        - High (60-100): Both corner lift AND squinch (genuine Duchenne smile)
        
        Returns raw 50-100; scaling (v-50)*2 yields display 0-100.
        """
        ear = cur.get("ear", 0.2)
        face_scale = max(1e-6, cur.get("face_scale", 50.0))
        mouth_pts = _safe(lm, MOUTH)
        baseline_ear = self._baseline_ear if self._baseline_ear > 0 else 0.25
        
        if len(mouth_pts) < 6:
            return 50.0
        
        # === AU 12: Lip Corner Pull (0-1 intensity) ===
        ly = float(mouth_pts[0, 1])
        ry = float(mouth_pts[5, 1]) if len(mouth_pts) > 5 else ly
        upper_pts = mouth_pts[1:min(5, len(mouth_pts)), :]
        upper_y = float(np.mean(upper_pts[:, 1])) if len(upper_pts) > 0 else ly
        corner_lift = upper_y - (ly + ry) / 2.0
        lift_ratio = corner_lift / face_scale
        
        AU12_threshold = 0.008
        au12_intensity = 0.0
        if lift_ratio > AU12_threshold:
            au12_intensity = min(1.0, (lift_ratio - AU12_threshold) / 0.05)
        
        # === AU 6: Cheek Raise / Eye Squinch (0-1 intensity) ===
        au6_intensity = 0.0
        if baseline_ear > 0.05:
            ear_ratio = ear / baseline_ear
            if 0.78 <= ear_ratio <= 0.96:
                au6_intensity = 1.0 - abs(ear_ratio - 0.87) / 0.18
                au6_intensity = max(0.0, min(1.0, au6_intensity))
        
        # === Combine: neither=low, one=medium, both=high ===
        # Each signal contributes up to 25 raw points; synergy adds up to 25 when both present
        corner_contrib = au12_intensity * 25.0
        squinch_contrib = au6_intensity * 25.0
        synergy = 0.0
        if au12_intensity > 0.2 and au6_intensity > 0.2:
            synergy = min(25.0, au12_intensity * au6_intensity * 30.0)
        
        raw_score = 50.0 + corner_contrib + squinch_contrib + synergy
        return float(max(50.0, min(100.0, raw_score)))

    def _g1_pupil_dilation(self, cur: dict, buf: list) -> float:
        """
        Pupil dilation proxy via eye openness vs baseline.
        Research: Hess (1965)—pupil dilation correlates with interest/arousal. We use
        eye-area ratio (wider eyes = higher) as proxy since pupil measurement unavailable.
        
        FALSE POSITIVE REDUCTION: Uses median of recent eye-area values (excluding blinks)
        to avoid single-frame noise. Requires sustained change from baseline.
        """
        is_blink = cur.get("is_blink", 0.0)
        area = cur.get("eye_area", 0.0)
        base = max(1e-6, self._baseline_eye_area)
        
        if is_blink > 0.5:
            self._last_pupil_dilation_score = 0.0
            return 0.0
        
        # TEMPORAL: Use median of recent eye areas (exclude blinks)
        if len(buf) >= 3:
            recent_areas = [b.get("eye_area", area) for b in buf[-4:]
                           if b.get("is_blink", 0) < 0.5]
            if recent_areas:
                area = float(np.median(recent_areas))
        
        r = area / base if base > 0 else 1.0
        
        # Stricter thresholds: require larger deviation from baseline
        if r >= 1.12:  # Strong dilation (raised from 1.08)
            score = 65.0 + min(30.0, (r - 1.12) * 250.0)
        elif r >= 1.06:
            score = 54.0 + (r - 1.06) / 0.06 * 10.0
        elif r >= 1.02:
            score = 50.0 + (r - 1.02) / 0.04 * 4.0
        elif r >= 0.98:
            score = 48.0 + (r - 0.98) / 0.04 * 2.0  # Near baseline = neutral
        elif r >= 0.92:
            score = 40.0 + (r - 0.92) / 0.06 * 8.0
        else:
            score = max(0.0, 40.0 + (r - 0.92) * 300.0)
        
        self._last_pupil_dilation_score = score
        return float(max(0.0, min(100.0, score)))

    def _g1_eyebrow_flash(self, buf: list) -> float:
        """
        Eyebrow flash: rapid bilateral raise (~200ms). Research: Ekman; cross-cultural
        signal of recognition, openness, interest. During conversation = nonverbal "yes."
        
        FALSE POSITIVE REDUCTION: Requires BOTH clear raise AND return pattern to
        distinguish from head movements or noise. Uses stricter threshold for raise.
        """
        if len(buf) < 10:
            return 50.0  # Warmup
        heights = [(b["eyebrow_l"] + b["eyebrow_r"]) / 2 for b in buf]
        face_scale = buf[-1].get("face_scale", 50.0)
        threshold = face_scale * 0.028  # Stricter (raised from 0.022)
        
        # Use first half as baseline (stable reference)
        baseline = float(np.median(heights[:max(2, len(heights) // 2)]))  # Median for robustness
        cur = heights[-1]
        recent_heights = heights[-5:]
        max_recent = float(np.max(recent_heights))
        
        raised = max_recent > baseline + threshold
        # Returning: current is clearly below peak (completed flash)
        returning = cur < max_recent - threshold * 0.5
        
        # Require BOTH raise AND return for high score (complete flash pattern)
        if raised and returning:
            flash_magnitude = (max_recent - baseline) / max(face_scale * 0.12, 1e-6)
            return 58.0 + min(32.0, flash_magnitude * 100.0)
        elif raised and cur > baseline + threshold * 0.6:  # Still raised
            raise_magnitude = (cur - baseline) / max(face_scale * 0.12, 1e-6)
            return 50.0 + min(20.0, raise_magnitude * 55.0)
        # Small sustained raise (not a flash, but mild interest)
        if cur > baseline + threshold * 0.6:
            return 50.0 + min(6.0, (cur - baseline) / max(face_scale * 0.10, 1e-6) * 20.0)
        return 50.0

    def _g1_eye_contact(self, cur: dict, w: int, h: int, buf: list) -> float:
        """
        Sustained eye contact: head orientation + face position.
        Research: eye-mind hypothesis; shared signal hypothesis; direct gaze = engagement.
        Social inclusion delays disengagement from direct gaze. Sustained facing = active listening.
        Uses median of recent yaw/pitch for robustness.
        """
        gx = cur.get("gaze_x", 0)
        gy = cur.get("gaze_y", 0)
        yaw = _median_recent(buf, "yaw", 3, cur.get("yaw", 0))
        pitch = _median_recent(buf, "pitch", 3, cur.get("pitch", 0))
        face_scale = cur.get("face_scale", 50.0)

        # Primary: head orientation toward camera (yaw/pitch near 0 = looking at camera)
        # Score 100 when head is forward; decay by angle (wider tolerance so normal movement still scores high)
        yaw_deg = abs(float(yaw))
        pitch_deg = abs(float(pitch))
        # 0° = 100; ~25° yaw or ~20° pitch = 50; 50°+ = 0
        head_score = 100.0 - min(100.0, (yaw_deg / 25.0) * 50.0 + (pitch_deg / 20.0) * 50.0)

        # Secondary: face position in frame (eye center vs frame center); cap effect so head dominates
        gaze_dist = np.sqrt(float(gx) * float(gx) + float(gy) * float(gy))
        gaze_normalized = min(1.0, gaze_dist / max(face_scale * 0.45, 1e-6))  # 45% = neutral; less strict
        gaze_bonus = (1.0 - gaze_normalized) * 15.0  # Up to +15 when face centered

        base_score = min(100.0, head_score + gaze_bonus)

        if len(buf) < 2:
            return float(max(0.0, min(100.0, base_score)))

        # Same formula for each frame in history (consistent with base_score)
        def _frame_eye_contact(frame: dict) -> float:
            gxf = frame.get("gaze_x", 0)
            gyf = frame.get("gaze_y", 0)
            yaw_f = abs(float(frame.get("yaw", 0)))
            pitch_f = abs(float(frame.get("pitch", 0)))
            fs = max(face_scale, frame.get("face_scale", 50.0), 20.0)
            head_s = 100.0 - min(100.0, (yaw_f / 25.0) * 50.0 + (pitch_f / 20.0) * 50.0)
            gd = np.sqrt(float(gxf) * float(gxf) + float(gyf) * float(gyf))
            gn = min(1.0, gd / max(fs * 0.45, 1e-6))
            g_bonus = (1.0 - gn) * 15.0
            return min(100.0, head_s + g_bonus)

        recent_frames = buf[-18:] if len(buf) >= 18 else buf
        frame_scores = [_frame_eye_contact(f) for f in recent_frames]

        # Count consecutive frames with good eye contact from current backward (lower threshold = more sustained)
        good_threshold = 50.0
        sustained_period = 0
        for s in reversed(frame_scores):
            if s > good_threshold:
                sustained_period += 1
            else:
                break

        avg_recent = float(np.mean(frame_scores)) if frame_scores else base_score

        if sustained_period >= 14:
            sustained_bonus = 28.0
        elif sustained_period >= 10:
            sustained_bonus = 18.0 + (sustained_period - 10) * 2.5
        elif sustained_period >= 6:
            sustained_bonus = 8.0 + (sustained_period - 6) * 2.5
        elif sustained_period >= 3:
            sustained_bonus = 2.0 + (sustained_period - 3) * 2.0
        else:
            sustained_bonus = 0.0

        consistency_bonus = min(8.0, max(0.0, (avg_recent - 55.0) / 45.0 * 8.0)) if avg_recent > 55.0 else 0.0

        final_score = base_score + sustained_bonus + consistency_bonus
        return float(max(0.0, min(100.0, final_score)))

    def _g1_head_tilt(self, cur: dict, buf: list) -> float:
        """
        Head tilt (lateral/roll): engagement signal from psychology research.
        
        Research basis (Davidenko et al., 2018; UC Santa Cruz / Perception):
        - Lateral tilt as small as 11° facilitates social engagement; makes faces more approachable.
        - Tilt exposes neck (trust/vulnerability); signals active listening vs passive.
        - Optimal band ~6-18°; extreme (>35°) can indicate confusion or unnatural pose.
        
        FALSE POSITIVE REDUCTION: Requires sustained tilt (checked via buffer) to avoid
        scoring high on momentary head movements or landmark noise.
        
        Returns raw 50-100; 50 = no tilt (display 0), high = tilt present (engagement).
        """
        roll = abs(float(cur.get("roll", 0)))
        
        # TEMPORAL CHECK: Use median of recent rolls to avoid single-frame spikes
        if len(buf) >= 4:
            recent_rolls = [abs(float(b.get("roll", 0))) for b in buf[-4:]]
            roll = float(np.median(recent_rolls))
        
        # No tilt (< 3.5°): neutral/passive; raw 50 -> display 0 (raised from 2.5)
        if roll < 3.5:
            return 50.0
        # Subtle tilt (3.5-7°): onset of interest
        if 3.5 <= roll < 7.0:
            return 50.0 + (roll - 3.5) / 3.5 * 14.0  # 50->64
        # Optimal band (7-18°): engagement/curiosity peak (Davidenko: 11° meaningful)
        if 7.0 <= roll <= 18.0:
            peak = 11.0
            dist = abs(roll - peak)
            return 68.0 + (1.0 - min(1.0, dist / 6.0)) * 24.0  # 68-92, peak at 11°
        # Strong tilt (18-28°): clear engagement, ramp down
        if 18.0 < roll <= 28.0:
            return 64.0 - (roll - 18.0) * 1.4  # 64->50
        # Very strong (28-40°): possible confusion; low-mid
        if 28.0 < roll <= 40.0:
            return 50.0 - (roll - 28.0) * 0.3  # 50->46
        # Extreme (> 40°): "too weird" per research
        return 45.0

    def _g1_forward_lean(self, cur: dict, buf: list) -> float:
        """
        Forward lean: approach motivation proxy. Research: Riskind & Gotay; embodied
        approach—leaning forward increases left frontal activation to appetitive stimuli,
        amplifies approach motivation. Lean toward = desire, interest, engagement.
        
        FALSE POSITIVE REDUCTION: Uses baseline and requires sustained lean (median of
        recent frames) to avoid scoring on momentary movements or noise.
        """
        z = cur.get("face_z", 0)
        base = self._baseline_z
        if base == 0 or z == 0:
            return 50.0
        
        # TEMPORAL: Use median of recent Z values for stability
        if len(buf) >= 4:
            recent_z = [b.get("face_z", z) for b in buf[-4:]]
            z = float(np.median(recent_z))
        
        # z < baseline = lean toward camera; smaller ratio = stronger lean
        # Require at least 2% change to avoid noise (raised from 1%)
        if z < base * 0.98:
            return 50.0 + min(48.0, (1.0 - z / base) * 180.0)  # Reduced sensitivity
        return 50.0

    def _g1_facial_symmetry(self, lm: np.ndarray, buf: list) -> float:
        """
        Facial Symmetry: Measures bilateral symmetry of facial features along vertical center line.
        
        High symmetry (balanced facial expression) indicates focused engagement.
        Low symmetry (asymmetric expression) may indicate distraction, discomfort, or lack of focus.
        
        Calculation:
        - Checks symmetry of eyes, eyebrows, mouth, and nose alignment
        - Mirrors right side features across face center (vertical midline) and compares to left
        - Measures both horizontal (75% weight) and vertical (25% weight) symmetry
        - Normalizes errors by face scale (inter-ocular distance) for size invariance
        - Uses realistic scoring curve: <3% error = 90-100, 3-6% = 70-90, 6-10% = 50-70, etc.
        - Accounts for natural facial asymmetry (most faces have 3-8% natural asymmetry)
        """
        if len(buf) < 2 or lm.shape[0] < 20:
            return 50.0
        
        face_scale = buf[-1].get("face_scale", 50.0)
        
        # Extract key facial features
        le = _safe(lm, LEFT_EYE)
        re = _safe(lm, RIGHT_EYE)
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        
        if len(le) < 3 or len(re) < 3:
            return 50.0
        
        # Calculate feature centers
        left_eye_center = np.mean(le[:, :2], axis=0)
        right_eye_center = np.mean(re[:, :2], axis=0)
        
        # Eyebrow centers (if available)
        left_brow_center = np.mean(lb[:, :2], axis=0) if len(lb) >= 2 else left_eye_center + np.array([0, -10])
        right_brow_center = np.mean(rb[:, :2], axis=0) if len(rb) >= 2 else right_eye_center + np.array([0, -10])
        
        # Mouth corners
        ml = lm[MOUTH_LEFT, :2] if lm.shape[0] > MOUTH_LEFT else np.zeros(2)
        mr = lm[MOUTH_RIGHT, :2] if lm.shape[0] > MOUTH_RIGHT else np.zeros(2)
        
        # Nose tip for vertical alignment check
        nose_tip = lm[NOSE_TIP, :2] if lm.shape[0] > NOSE_TIP else (left_eye_center + right_eye_center) / 2.0
        
        # Calculate face center (vertical midline)
        face_center_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
        
        # Mirror right side features across center and compare to left
        # For perfect symmetry: left_x = 2*center_x - right_x
        right_eye_mirrored_x = 2 * face_center_x - right_eye_center[0]
        right_brow_mirrored_x = 2 * face_center_x - right_brow_center[0]
        right_mouth_mirrored_x = 2 * face_center_x - mr[0]
        
        # Calculate symmetry errors in pixels first
        eye_symmetry_error_x_px = abs(left_eye_center[0] - right_eye_mirrored_x)
        brow_symmetry_error_x_px = abs(left_brow_center[0] - right_brow_mirrored_x)
        mouth_symmetry_error_x_px = abs(ml[0] - right_mouth_mirrored_x)
        
        # Vertical alignment check (nose should be centered between eyes)
        eye_midpoint_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
        nose_alignment_error_px = abs(nose_tip[0] - eye_midpoint_x)
        
        # Vertical symmetry: check if features are at similar heights
        eye_height_diff_px = abs(left_eye_center[1] - right_eye_center[1])
        brow_height_diff_px = abs(left_brow_center[1] - right_brow_center[1])
        mouth_height_diff_px = abs(ml[1] - mr[1])
        
        # Normalize errors by face scale (inter-ocular distance)
        # This makes the metric invariant to face size and distance from camera
        face_scale_safe = max(face_scale, 20.0)  # Ensure minimum scale
        
        eye_symmetry_error_x = eye_symmetry_error_x_px / face_scale_safe
        brow_symmetry_error_x = brow_symmetry_error_x_px / face_scale_safe
        mouth_symmetry_error_x = mouth_symmetry_error_x_px / face_scale_safe
        nose_alignment_error = nose_alignment_error_px / face_scale_safe
        
        eye_height_diff = eye_height_diff_px / face_scale_safe
        brow_height_diff = brow_height_diff_px / face_scale_safe
        mouth_height_diff = mouth_height_diff_px / face_scale_safe
        
        # Combine all symmetry errors with weights
        # Eyes are most important (40%), then mouth (30%), brows (20%), nose (10%)
        horizontal_error = (
            0.4 * eye_symmetry_error_x +
            0.3 * mouth_symmetry_error_x +
            0.2 * brow_symmetry_error_x +
            0.1 * nose_alignment_error
        )
        
        # Vertical symmetry (less weight, as natural faces have some vertical asymmetry)
        vertical_error = (
            0.5 * eye_height_diff +
            0.3 * mouth_height_diff +
            0.2 * brow_height_diff
        )
        
        # Combined error (horizontal is more important for symmetry perception)
        total_error = 0.75 * horizontal_error + 0.25 * vertical_error
        
        # Convert error to score using a realistic curve that spans 0-100
        # Real faces typically have 3-8% natural asymmetry
        # Perfect symmetry (0% error) = 100
        # Excellent symmetry (< 3% error) = 90-100
        # Good symmetry (3-6% error) = 70-90
        # Moderate symmetry (6-10% error) = 50-70
        # Poor symmetry (10-15% error) = 30-50
        # Very poor symmetry (> 15% error) = 0-30
        
        # Finer bands: small asymmetry changes produce larger score changes
        if total_error < 0.02:
            symmetry_score = 100.0 - (total_error / 0.02) * 8.0
        elif total_error < 0.045:
            symmetry_score = 92.0 - ((total_error - 0.02) / 0.025) * 22.0
        elif total_error < 0.08:
            symmetry_score = 70.0 - ((total_error - 0.045) / 0.035) * 22.0
        elif total_error < 0.12:
            symmetry_score = 48.0 - ((total_error - 0.08) / 0.04) * 20.0
        else:
            capped_error = min(total_error, 0.28)
            symmetry_score = max(0.0, 28.0 - ((capped_error - 0.12) / 0.16) * 28.0)
        
        return float(max(0.0, min(100.0, symmetry_score)))

    def _g1_rhythmic_nodding(self, buf: list) -> float:
        """
        Rhythmic nodding: vertical pitch oscillations. Research: Wells & Petty (1980)—
        nodding increases persuasion (self-validation). Nodding = agreement, engagement.
        We detect pitch oscillations (zero-crossings) as nodding rhythm.
        """
        if len(buf) < 12:
            return 40.0
        pitches = [b["pitch"] for b in buf[-12:]]
        pitch_std = float(np.std(pitches))
        if pitch_std < 0.6:
            return 34.0
        # Simple: 1–2 “oscillations” in pitch
        d = np.diff(pitches)
        crosses = np.sum((d[:-1] * d[1:]) < 0)
        if 1 <= crosses <= 4:
            return 68.0 + min(12.0, pitch_std * 4.0)
        if pitch_std > 1.2:
            return 48.0
        return 42.0

    def _g1_parted_lips(self, cur: dict, buf: list) -> float:
        """
        Parted lips: slight mouth opening. Research: open mouth = receptivity, listening,
        preparation to speak. Closed = withholding; parted = engaged, attentive.
        
        FALSE POSITIVE REDUCTION: Uses median of recent MAR values to avoid triggering
        on speech artifacts or single-frame noise.
        """
        mar = cur.get("mar", 0.2)
        
        # TEMPORAL: Use median of recent MAR for stability (avoid speech artifacts)
        if len(buf) >= 3:
            recent_mar = [b.get("mar", mar) for b in buf[-3:]]
            mar = float(np.median(recent_mar))
        
        # Narrower optimal band to avoid false positives
        if 0.12 <= mar <= 0.32:
            return 48.0 + (mar - 0.12) / 0.20 * 42.0  # 48-90
        if 0.08 <= mar < 0.12:
            return 40.0 + (mar - 0.08) / 0.04 * 8.0
        if 0.32 < mar <= 0.48:
            return 80.0 - (mar - 0.32) / 0.16 * 26.0  # Ramp down for wide open
        return 38.0 + min(8.0, mar * 20.0)

    def _g1_softened_forehead(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        """
        Softened forehead: relaxed brow (low tension). Research: FACS AU 4 (brow lowerer)
        absent = relaxed. Low brow variance + even inner/outer = not furrowed = relaxed engagement.
        
        FALSE POSITIVE REDUCTION: Uses median of recent brow measurements and requires
        sustained relaxation pattern to distinguish from momentary expressions.
        """
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        if len(lb) < 2 or len(rb) < 2:
            return 50.0
        face_scale = max(20.0, cur.get("face_scale", 50.0))

        # 1) Variance of brow points (low = flat = relaxed)
        v = float(np.var(np.vstack([lb[:, :2], rb[:, :2]])))
        v_normalized = v / max(face_scale * face_scale, 1e-6)

        # 2) Inner vs outer brow height: relaxed = similar height; furrowed = inner pulled down
        left_inner_y = float(lb[np.argmin(lb[:, 0]), 1])
        left_outer_y = float(lb[np.argmax(lb[:, 0]), 1])
        right_inner_y = float(rb[np.argmin(rb[:, 0]), 1])
        right_outer_y = float(rb[np.argmax(rb[:, 0]), 1])
        inner_outer_diff_left = left_inner_y - left_outer_y
        inner_outer_diff_right = right_inner_y - right_outer_y
        diff_normalized = (abs(inner_outer_diff_left) + abs(inner_outer_diff_right)) / max(face_scale * 0.15, 1e-6)
        evenness = max(0.0, 1.0 - min(1.0, diff_normalized))

        # Azure-friendly: when only 2 points per brow (4 unique), use flat-brow metric from evenness only
        brow_pts = np.vstack([lb[:, :2], rb[:, :2]])
        unique_rows = np.unique(np.round(brow_pts).astype(np.int32), axis=0)
        few_distinct_brow_points = len(unique_rows) <= 4

        # TEMPORAL: Check consistency over recent frames for stability
        sustained_relaxation = True
        if len(buf) >= 4:
            # Check if brow has been consistently even (not temporarily relaxed)
            recent_bl = [b.get("eyebrow_l", 0) for b in buf[-4:]]
            recent_br = [b.get("eyebrow_r", 0) for b in buf[-4:]]
            if recent_bl and recent_br:
                brow_std = float(np.std(recent_bl)) + float(np.std(recent_br))
                # Relaxed threshold so normal jitter doesn't disable sustained (was 0.04)
                if brow_std > face_scale * 0.08:
                    sustained_relaxation = False

        # Score from variance: low v_norm -> high (relaxed)
        # Azure path: 2 points per brow (4 unique) -> use evenness only so we don't penalize low point count
        if few_distinct_brow_points:
            var_score = 65.0 + 17.0 * evenness
        else:
            # Relaxed bands so typical/Azure-expanded brows get 60–82 range (was 0.0006/0.0025/0.006)
            if v_normalized < 0.002 and sustained_relaxation:
                var_score = 82.0
            elif v_normalized < 0.008:
                var_score = 68.0 + (0.008 - v_normalized) / 0.006 * 12.0
            elif v_normalized < 0.018:
                var_score = 52.0 + (0.018 - v_normalized) / 0.010 * 14.0
            else:
                # Gentler decay and higher floor so moderate variance still scores above 48
                var_score = max(44.0, 58.0 - (v_normalized - 0.018) * 400.0)

        # Combine: 60% variance (flat brow), 40% evenness
        raw = 50.0 + 0.5 * (var_score - 50.0) + 15.0 * evenness
        # Soft penalty when not sustained so jitter doesn't force score below 48 (was 0.85)
        if not sustained_relaxation:
            raw = raw * 0.92
        return float(max(32.0, min(90.0, raw)))

    # ----- Group 2 -----
    def _g2_look_up_lr(self, cur: dict, buf: list) -> float:
        """
        Look up/left/right: cognitive load cue. Research: gaze aversion during difficult
        tasks; look-up-left (NLU) = visual/constructed imagery. Eye-mind link: gaze
        reflects cognitive processing. Upward/lateral = thinking, accessing memory.
        
        FALSE POSITIVE REDUCTION: Uses median of recent pitch/yaw and requires
        sustained gaze shift (not momentary glance). Stricter thresholds.
        """
        p = float(cur.get("pitch", 0))
        y = float(cur.get("yaw", 0))
        
        # TEMPORAL: Use median of recent values for stability
        if len(buf) >= 3:
            recent_pitch = [b.get("pitch", p) for b in buf[-3:]]
            recent_yaw = [b.get("yaw", y) for b in buf[-3:]]
            p = float(np.median(recent_pitch))
            y = float(np.median(recent_yaw))
        
        # Stricter thresholds: require clearer gaze shifts
        looking_up = p > 5  # Raised from 3
        looking_lr = abs(y) > 8  # Raised from 6
        
        if looking_up and looking_lr:
            up_intensity = min(1.0, (p - 5) / 18.0)
            lr_intensity = min(1.0, (abs(y) - 8) / 28.0)
            combined = (up_intensity + lr_intensity) / 2.0
            return 52.0 + combined * 38.0
        elif looking_up:
            up_intensity = min(1.0, (p - 5) / 12.0)
            return 50.0 + up_intensity * 28.0
        elif looking_lr:
            lr_intensity = min(1.0, (abs(y) - 8) / 30.0)
            return 48.0 + lr_intensity * 16.0
        return 45.0

    def _g2_lip_pucker(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        """
        Lip Pucker: Detects pursed lips (thinking, evaluating expression).
        
        Lip pucker is characterized by:
        - High MAR (mouth aspect ratio - vertical opening relative to width)
        - Narrow mouth width (lips pursed together)
        - Both conditions must be met for a pucker
        
        FALSE POSITIVE REDUCTION: Uses median of recent MAR/mouth width to avoid
        triggering on speech artifacts. Requires sustained pucker pattern.
        
        Returns 0-100 where:
        - 0-30: No pucker (normal mouth)
        - 30-60: Slight pucker
        - 60-80: Moderate pucker
        - 80-100: Strong pucker
        """
        mar = cur.get("mar", 0.2)
        face_scale = cur.get("face_scale", 50.0)
        mw = cur.get("mouth_w", 40.0)
        
        # TEMPORAL: Use median of recent values to avoid speech artifacts
        if len(buf) >= 3:
            recent_mar = [b.get("mar", mar) for b in buf[-3:]]
            recent_mw = [b.get("mouth_w", mw) for b in buf[-3:]]
            mar = float(np.median(recent_mar))
            mw = float(np.median(recent_mw))
        
        # Normalize mouth width by face scale for size invariance
        mw_normalized = mw / max(face_scale, 1e-6)
        
        # Normal mouth characteristics:
        # - MAR typically 0.15-0.25 for relaxed/neutral mouth
        # - Mouth width normalized: typically 0.20-0.35 of face scale
        
        # Lip pucker requires BOTH:
        # 1. High MAR (vertical opening) - indicates lips are pursed forward
        # 2. Narrow mouth width - indicates lips are compressed horizontally
        
        # No pucker: normal MAR and normal width (lower MAR threshold for sensitivity)
        if mar <= 0.22 and mw_normalized >= 0.19:
            return 10.0 + (mar / 0.22) * 12.0
        if mar <= 0.28:
            # MAR not high enough for pucker
            if mw_normalized < 0.15:
                # Narrow but not puckered (might be speaking or other expression)
                return 20.0 + (0.30 - mar) / 0.05 * 10.0  # 20-30
            else:
                # Normal mouth
                return 5.0 + (mar / 0.30) * 15.0  # 5-20
        
        if mw_normalized >= 0.20:
            return 15.0 + min(18.0, (mar - 0.28) / 0.08 * 18.0)
        
        # High MAR and narrow mouth = pucker (lower MAR threshold 0.28)
        if mar > 0.38 and mw_normalized < 0.12:
            pucker_intensity = min(1.0, (mar - 0.38) / 0.14)
            return 86.0 + pucker_intensity * 14.0
        elif mar > 0.32 and mw_normalized < 0.15:
            pucker_intensity = (mar - 0.32) / 0.06
            return 70.0 + pucker_intensity * 16.0
        elif mar > 0.28 and mw_normalized < 0.18:
            mar_factor = (mar - 0.28) / 0.04
            width_factor = (0.18 - mw_normalized) / 0.06
            pucker_intensity = (mar_factor + width_factor) / 2.0
            return 50.0 + pucker_intensity * 24.0
        else:
            mar_contribution = max(0.0, (mar - 0.22) / 0.10)
            width_contribution = max(0.0, (0.20 - mw_normalized) / 0.08)
            combined = (mar_contribution + width_contribution) / 2.0
            return 28.0 + combined * 24.0

    def _g2_eye_squint(self, cur: dict, buf: list) -> float:
        """
        Eye squint: narrowed eyes. Research: FACS AU 7; squinting = skepticism, distrust,
        evaluation. Combined with pursed lips = doubt. Cognitive load increases squinting.
        
        FALSE POSITIVE REDUCTION: Compares to baseline EAR and uses median of recent
        frames to avoid triggering on blinks or single-frame noise.
        """
        ear = cur.get("ear", 0.2)
        baseline_ear = self._baseline_ear if self._baseline_ear > 0.05 else 0.22
        
        # TEMPORAL: Use median of recent EAR for stability
        if len(buf) >= 3:
            recent_ear = [b.get("ear", ear) for b in buf[-3:] if b.get("is_blink", 0) < 0.5]
            if recent_ear:
                ear = float(np.median(recent_ear))
        
        # Compare to baseline: squint = EAR below baseline
        ear_ratio = ear / baseline_ear if baseline_ear > 0 else 1.0
        
        # Only score high when clearly below baseline (not just naturally narrow eyes)
        if ear_ratio < 0.75 and ear < 0.15:  # Strong squint
            return 68.0 + min(24.0, (0.75 - ear_ratio) * 120.0)
        if ear_ratio < 0.85 and ear < 0.17:  # Moderate squint
            return 52.0 + (0.85 - ear_ratio) * 160.0
        if ear_ratio < 0.92:  # Slight squint
            return 42.0 + (0.92 - ear_ratio) * 100.0
        return 38.0  # Normal eyes

    def _g2_thinking_brow(self, lm: np.ndarray, buf: list) -> float:
        """
        Thinking brow: asymmetric brow raise. Research: one brow raised = curiosity,
        skepticism, or concentration. Different from bilateral flash; sustained asymmetry
        = evaluation, doubt.
        
        FALSE POSITIVE REDUCTION: Requires sustained asymmetry (not single-frame) and
        meaningful threshold relative to face scale. Natural brow asymmetry filtered.
        """
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        if len(lb) < 2 or len(rb) < 2:
            return 45.0
        ly, ry = float(np.mean(lb[:, 1])), float(np.mean(rb[:, 1]))
        d = abs(ly - ry)
        
        # Get face scale from buffer if available
        if len(buf) > 0:
            face_scale = buf[-1].get("face_scale", 50.0)
        else:
            face_scale = float(np.max(lm[:, 0]) - np.min(lm[:, 0])) * 0.5 if lm.shape[0] > 0 else 50.0
        face_scale = max(20.0, face_scale)
        d_rel = d / face_scale
        
        # TEMPORAL: Check if asymmetry is sustained (not just single-frame noise)
        if len(buf) >= 3:
            recent_asymmetries = []
            for b in buf[-3:]:
                # Estimate brow asymmetry from eyebrow heights stored in buffer
                bl = b.get("eyebrow_l", 0)
                br = b.get("eyebrow_r", 0)
                fs = b.get("face_scale", 50.0)
                if fs > 0:
                    recent_asymmetries.append(abs(bl - br) / max(fs, 20.0))
            # Require sustained asymmetry (at least 2 of 3 frames)
            high_asymmetry_count = sum(1 for a in recent_asymmetries if a > 0.025)
            if high_asymmetry_count < 2:
                return 44.0
        
        # Stricter thresholds: require meaningful asymmetry
        if 0.04 <= d_rel <= 0.18:  # Clear asymmetry
            return 68.0 + min(14.0, (d_rel - 0.04) * 100.0)
        if 0.025 <= d_rel < 0.04:  # Slight asymmetry
            return 52.0 + (d_rel - 0.025) * 400.0
        return 44.0  # Natural/no asymmetry

    def _g2_stillness(self, buf: list) -> float:
        """
        Stillness: low facial movement. Research: can indicate focused attention or
        frozen/withdrawn state. Low face_var = minimal landmark movement. Context-dependent.
        
        FALSE POSITIVE REDUCTION: Stillness is ambiguous (could be focused listening
        or frozen state). Use more conservative scoring to avoid flagging attentive
        listeners. Require very low variance for extended period.
        """
        if len(buf) < 10:
            return 50.0  # Warmup
        vars_list = [b["face_var"] for b in buf[-10:]]
        scales = [b.get("face_scale", 50.0) for b in buf[-10:]]
        avg_scale = float(np.mean(scales)) if scales else 50.0
        avg_scale_sq = max(400.0, avg_scale * avg_scale)
        m = float(np.median(vars_list))  # Median: robust to single-frame movement spikes
        m_normalized = m / avg_scale_sq
        
        # Stricter: require VERY low variance for stillness (attentive listening is normal)
        if m_normalized < 0.0008:  # Very strict (extremely still)
            return 68.0
        if m_normalized < 0.0025:  # Moderately still
            return 55.0 + (0.0025 - m_normalized) / 0.0017 * 12.0
        if m_normalized < 0.005:  # Slightly still
            return 46.0 + (0.005 - m_normalized) / 0.0025 * 8.0
        return 42.0  # Normal movement

    def _g2_lowered_brow(self, lm: np.ndarray, buf: list) -> float:
        """
        Lowered brow (furrowed): FACS AU 4. Research: corrugator activation = concentration,
        cognitive load, OR frustration/anger. CONTEXT-DEPENDENT: AU4 can signal (1) effortful
        thinking (G2), (2) problem understanding (communicative signal), or (3) negative
        affect (G3). This metric is in G2 (cognitive load); interpret in combination with
        other G2/G3 signals. Furrowed = thinking hard or negative affect.
        
        FALSE POSITIVE REDUCTION: Compares to recent baseline brow position (not absolute)
        and requires sustained lowering. Natural brow position varies per person.
        """
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        le = _safe(lm, LEFT_EYE)
        re = _safe(lm, RIGHT_EYE)
        if len(lb) < 2 or len(le) < 2:
            return 46.0
        
        brow_y = (float(np.mean(lb[:, 1])) + float(np.mean(rb[:, 1]))) / 2
        eye_y = (float(np.mean(le[:, 1])) + float(np.mean(re[:, 1]))) / 2
        dist = eye_y - brow_y  # Smaller distance = furrowed (brow lowered)
        
        # Get face scale
        if len(buf) > 0:
            face_scale = buf[-1].get("face_scale", 50.0)
        else:
            face_scale = float(np.max(lm[:, 0]) - np.min(lm[:, 0])) * 0.5 if lm.shape[0] > 0 else 50.0
        face_scale = max(20.0, face_scale)
        dist_rel = dist / face_scale
        
        # BASELINE: Estimate person's typical brow-eye distance from history
        baseline_dist_rel = dist_rel
        if len(buf) >= 8:
            # Use median of first half of buffer as baseline
            half = max(4, len(buf) // 2)
            baseline_dists = []
            for b in buf[:half]:
                bl = b.get("eyebrow_l", 0)
                br = b.get("eyebrow_r", 0)
                fs = b.get("face_scale", 50.0)
                # Approximate: eyebrow height stored; lower = furrowed
                if fs > 0:
                    baseline_dists.append((bl + br) / (2 * max(fs, 20.0)))
            if baseline_dists:
                baseline_dist_rel = float(np.median(baseline_dists)) * 0.15 + 0.18  # Approximate
        
        # Compare to baseline: lower than typical = furrowed
        furrowed_amount = baseline_dist_rel - dist_rel
        
        # Stricter thresholds
        if dist_rel < 0.12 and furrowed_amount > 0.04:  # Very furrowed
            return 75.0 + min(15.0, furrowed_amount * 200.0)
        elif dist_rel < 0.16 and furrowed_amount > 0.02:  # Moderately furrowed
            return 58.0 + (0.16 - dist_rel) / 0.04 * 14.0
        elif dist_rel < 0.20:  # Slightly furrowed
            return 48.0 + (0.20 - dist_rel) / 0.04 * 8.0
        return 44.0

    # ----- Group 3 -----
    def _g3_contempt(
        self,
        lm: np.ndarray,
        ml: np.ndarray,
        mr: np.ndarray,
        fr: Optional[FaceDetectionResult],
        cur: dict,
        buf: list,
    ) -> float:
        """
        Contempt: unilateral lip curl (one corner raised). Research: Ekman & Friesen.
        CRITICAL: High false positive risk. Uses (1) baseline adaptation, (2) head-roll
        correction, (3) temporal consistency, (4) strict absolute thresholds.
        """
        # --- Azure path: require high confidence; scale gently ---
        if fr and getattr(fr, "emotions", None) and isinstance(fr.emotions, dict):
            c = float(fr.emotions.get("contempt", 0.0) or 0.0)
            # Only score above neutral when contempt > 0.55; scale gently to avoid persistent 100
            if c <= 0.45:
                return 35.0  # Below threshold = neutral/low
            if c <= 0.55:
                return 42.0 + (c - 0.45) * 80.0  # 42–50
            return min(82.0, 52.0 + (c - 0.55) * 80.0)  # 52–82 for c in [0.55, 0.92]

        # --- Landmark path: baseline + roll correction + temporal consistency ---
        if lm.shape[0] <= max(MOUTH_LEFT, MOUTH_RIGHT):
            return 35.0

        face_scale = cur.get("face_scale", 50.0)
        face_scale = max(20.0, face_scale)
        ly, ry = float(ml[1]), float(mr[1])
        raw_asymmetry_ratio = abs(ly - ry) / face_scale

        # Head-roll correction: tilt creates apparent asymmetry; subtract expected component
        roll = abs(float(cur.get("roll", 0)))
        roll_correction = 0.0
        if roll > 4.0:
            roll_correction = min(0.10, roll * 0.004)  # Up to 0.10 from ~25° roll
        corrected_ratio = max(0.0, raw_asymmetry_ratio - roll_correction)

        # Update history for baseline and temporal check
        self._contempt_asymmetry_history.append(corrected_ratio)

        # Baseline: median of last 18 frames (person's typical asymmetry when neutral)
        hist_list = list(self._contempt_asymmetry_history)
        if len(hist_list) < 10:
            return 35.0  # Warmup: need enough history for stable baseline

        baseline = float(np.median(hist_list[-18:])) if len(hist_list) >= 18 else float(np.median(hist_list))

        # Require (a) absolute threshold, (b) above baseline, (c) temporal consistency
        abs_threshold = 0.14  # Pronounced asymmetry only
        above_baseline = corrected_ratio > baseline + 0.06  # 6% above typical
        abs_met = corrected_ratio > abs_threshold

        # Temporal: at least 2 of last 3 frames must also meet criteria
        recent_ok = 0
        if len(buf) >= 3:
            for b in buf[-3:]:
                ar = b.get("mouth_corner_asymmetry_ratio", 0.0)
                roll_b = abs(float(b.get("roll", 0)))
                rc = min(0.10, roll_b * 0.004) if roll_b > 4.0 else 0.0
                cr = max(0.0, ar - rc)
                if cr > abs_threshold and cr > baseline + 0.05:
                    recent_ok += 1
        temporal_met = recent_ok >= 2

        if not (abs_met and above_baseline and temporal_met):
            return 35.0

        # Score: how far above baseline; cap so we need pronounced contempt for high score
        excess = corrected_ratio - baseline - 0.06
        raw_score = 50.0 + min(32.0, excess * 250.0)  # 50–82 range
        return float(max(50.0, min(82.0, raw_score)))

    def _g3_nose_crinkle(self, cur: dict, buf: list) -> float:
        """
        Nose crinkle: nose shortening. Research: FACS AU 9; wrinkling nose = disgust,
        skepticism. Levator labii activates; nose shortens. CAUTION: Can also indicate
        concentration or reaction to strong stimulus. Require SUSTAINED shortening
        (>3 frames) and LARGE drop (>8%) to reduce false positives.
        """
        nh = cur.get("nose_height", 0.0)
        if nh <= 0 or len(buf) < 6:
            return 8.0
        recent = [b.get("nose_height", nh) for b in buf[-6:-1] if b.get("nose_height", 0) > 0]
        if not recent:
            return 8.0
        avg = float(np.mean(recent))
        if avg <= 1e-6:
            return 8.0
        # Require LARGER shortening (8%+) to reduce false positives
        if nh < avg * 0.88:  # Stricter: 12% drop (was 0.92)
            return 52.0 + min(36.0, (1.0 - nh / avg) * 90.0)
        if nh < avg * 0.94:  # Moderate shortening (6%+)
            return 28.0 + (0.94 - nh / avg) * 200.0
        return 6.0  # Default low

    def _g3_lip_compression(self, cur: dict, buf: list) -> float:
        """
        Lip compression: FACS AU 23/24. Research: pursed/compressed lips = disapproval,
        emotional restraint, withholding. "Preventing critical thoughts from being spoken."
        CAUTION: Can also indicate concentration, controlled speech, or processing in
        professional contexts.

        FALSE POSITIVE REDUCTION: (1) Compare to baseline MAR—only score high when
        current MAR is clearly BELOW person's typical mouth openness. (2) Use median
        of recent 4 frames to avoid single-frame/speech artifacts. (3) Require
        temporal consistency: at least 3 of last 4 frames below threshold. (4) Stricter
        absolute thresholds so neutral/resting lips rarely trigger.
        """
        mar = cur.get("mar_inner", cur.get("mar", 0.2))
        baseline_mar = max(0.08, self._baseline_mar) if self._baseline_mar > 0 else 0.20

        # WARMUP: Require enough history for stable baseline; otherwise return low
        buf_len = len(buf)
        if buf_len < 18:
            return 4.0
        if baseline_mar < 0.06:
            return 4.0

        # TEMPORAL: Use median of last 6 frames (longer window = fewer speech artifacts)
        if buf_len >= 6:
            recent_mar = [b.get("mar_inner", b.get("mar", mar)) for b in buf[-6:]]
            mar = float(np.median(recent_mar))
        mar_ratio = mar / baseline_mar if baseline_mar > 0 else 1.0

        # SUSTAINED: Require 5 of last 6 frames below baseline*0.78 (stricter)
        sustained = False
        if buf_len >= 6:
            below_count = sum(
                1 for b in buf[-6:]
                if (b.get("mar_inner", b.get("mar", 0.2)) < baseline_mar * 0.78)
            )
            sustained = below_count >= 5

        # Only score above neutral when clearly below baseline (mar_ratio < 0.75)
        if mar_ratio >= 0.75:
            return 4.0
        if not sustained:
            if mar < 0.06 and mar_ratio < 0.85:
                return 8.0 + (0.75 - mar_ratio) * 20.0
            return 4.0

        # Stricter absolute bands; require sustained + below baseline
        if mar < 0.022 and mar_ratio < 0.68:
            return 68.0 + min(18.0, (0.68 - mar_ratio) * 50.0)
        if mar < 0.032 and mar_ratio < 0.72:
            return 52.0 + (0.032 - mar) / 0.010 * 14.0
        if mar < 0.045 and mar_ratio < 0.75:
            return 38.0 + (0.045 - mar) / 0.013 * 10.0
        if mar < 0.058 and mar_ratio < 0.80:
            return 24.0 + (0.058 - mar) / 0.013 * 10.0
        if mar < 0.075 and mar_ratio < 0.85:
            return 12.0 + (0.075 - mar) / 0.017 * 8.0
        return 4.0

    def _g3_eye_block(self, buf: list) -> float:
        """
        Eye block: prolonged closure. Research: shutting out, aversion. Extended
        eye closure (EAR < 0.1) = blocking visual input, disengagement, or distress.
        CAUTION: Normal blinks last 1-3 frames. Only score high when PROLONGED
        (>12 frames = ~400ms at 30fps) to avoid false positives from blinks.
        """
        if len(buf) < 6:
            return 6.0
        run = 0
        for b in reversed(buf):
            if b.get("ear", 0.2) < 0.10:
                run += 1
            else:
                break
        # Require LONGER closure to distinguish from blinks
        if run >= 18:  # ~600ms = definite block
            return 88.0
        if run >= 12:  # ~400ms = likely block (was 6)
            return 55.0 + (run - 12) / 6 * 28.0
        if run >= 8:  # ~267ms = possible block
            return 28.0 + (run - 8) * 4.0
        return 6.0  # Brief closure = likely blink

    def _g3_jaw_clench(self, lm: np.ndarray, cur: dict, fr: Optional[FaceDetectionResult], buf: list) -> float:
        """
        Jaw clench: tight jaw + mouth. Research: masseter tension = stress, resistance,
        suppressed aggression. Low MAR + corners down = clenched, tense.

        FALSE POSITIVE REDUCTION (minimal false detections): (1) Require BOTH tight
        lips AND pronounced corners-down (8% of face_scale). (2) Warmup 18+ frames;
        baseline comparison (mar_ratio < 0.72 for high score). (3) Sustained low MAR:
        5 of last 6 frames below baseline*0.72. (4) Compression alone (no corners down)
        capped at raw 36 so display never shows 100. Default 4.0.
        """
        mar = cur.get("mar_inner", cur.get("mar", 0.2))
        baseline_mar = max(0.08, self._baseline_mar) if self._baseline_mar > 0 else 0.20
        face_scale = cur.get("face_scale", 50.0)
        mouth_pts = _safe(lm, MOUTH)

        # WARMUP: Require enough history for stable baseline
        buf_len = len(buf)
        if buf_len < 18:
            return 4.0
        if baseline_mar < 0.06:
            return 4.0

        # TEMPORAL: Median of last 6 frames
        if buf_len >= 6:
            recent_mar = [b.get("mar_inner", b.get("mar", mar)) for b in buf[-6:]]
            mar = float(np.median(recent_mar))
        mar_ratio = mar / baseline_mar if baseline_mar > 0 else 1.0

        # Corners down: mouth corners noticeably BELOW lip midline (8% of face — very strict)
        corners_down = False
        if len(mouth_pts) >= 6:
            ly = float(mouth_pts[0, 1])
            ry = float(mouth_pts[5, 1])
            mid = float(np.mean(mouth_pts[:, 1]))
            threshold = face_scale * 0.08  # 8% — neutral mouths rarely qualify
            corners_down = (ly + ry) / 2 > mid + threshold

        # SUSTAINED: 5 of last 6 frames with MAR below baseline*0.72
        sustained_low_mar = False
        if buf_len >= 6:
            below_count = sum(
                1 for b in buf[-6:]
                if (b.get("mar_inner", b.get("mar", 0.2)) < baseline_mar * 0.72)
            )
            sustained_low_mar = below_count >= 5

        # High score ONLY when BOTH corners down AND sustained low MAR AND below baseline
        if mar < 0.038 and corners_down and mar_ratio < 0.72 and sustained_low_mar:
            return min(82.0, 68.0 + (0.038 - mar) * 200.0)
        if mar < 0.048 and corners_down and mar_ratio < 0.78 and sustained_low_mar:
            return 52.0 + (0.048 - mar) / 0.010 * 18.0
        if mar < 0.058 and corners_down and mar_ratio < 0.82:
            return 38.0 + (0.058 - mar) / 0.010 * 12.0
        if mar < 0.070 and corners_down:
            return 24.0 + (0.070 - mar) / 0.012 * 10.0
        # Compression alone (no corners down): cap at raw 36 so display stays 0
        if mar < 0.038 and mar_ratio < 0.72:
            return 32.0 + (0.038 - mar) / 0.010 * 4.0
        if mar < 0.055 and mar_ratio < 0.80:
            return 18.0 + (0.055 - mar) / 0.017 * 10.0
        if mar < 0.075:
            return 8.0 + (0.075 - mar) / 0.020 * 6.0
        return 4.0

    def _g3_rapid_blink(self) -> float:
        """
        Rapid blinking: elevated blink rate. Research: blink rate increases with stress,
        cognitive load, anxiety. CAUTION: Normal blink rate varies (10-20/min); only
        score high when ELEVATED (5+ blinks in window). Context: rapid blinking can
        also indicate concentration or dry eyes. Use in combination with other G3 signals.
        """
        b = self._blinks_in_window
        # Require higher count to reduce false positives
        if b >= 6:  # Raised from 5
            return 82.0
        if b >= 4:  # Raised from 3
            return 48.0 + (b - 4) * 15.0
        return 6.0  # Normal blink rate

    def _g3_gaze_aversion(self, cur: dict, buf: list) -> float:
        """
        Gaze aversion: looking away. Research: CRITICAL CONTEXT DEPENDENCY—gaze aversion
        serves DUAL functions: (1) internal cognitive processing/memory retrieval (positive),
        (2) disengagement/discomfort (negative). Research shows gaze-away occurs within 1s
        during memory retrieval, lasting ~6s, and is HIGHER for effortful retrieval.
        
        Professional context: off-camera gaze is judged negatively despite cognitive function.
        To avoid false positives: only score high when SUSTAINED (>8 frames) AND combined
        with other resistance signals. Brief gaze shifts (<5 frames) are likely cognitive
        processing, NOT resistance.
        """
        if len(buf) < 5:
            return 8.0
        
        p = cur.get("pitch", 0)
        y = cur.get("yaw", 0)
        
        # Combine pitch and yaw for total gaze deviation
        pitch_dev = abs(p)
        yaw_dev = abs(y)
        total_dev = np.sqrt(pitch_dev * pitch_dev + yaw_dev * yaw_dev)
        
        # Check DURATION of gaze aversion to distinguish processing from disengagement
        recent_devs = []
        for b in buf[-8:]:
            bp = abs(b.get("pitch", 0))
            by = abs(b.get("yaw", 0))
            recent_devs.append(np.sqrt(bp * bp + by * by))
        
        # Count consecutive frames with high deviation (sustained aversion)
        sustained_aversion = 0
        for d in reversed(recent_devs):
            if d > 8.0:  # Threshold for "averted"
                sustained_aversion += 1
            else:
                break
        
        # Only score high if SUSTAINED (>6 frames = likely disengagement, not brief processing)
        if sustained_aversion >= 8 and total_dev > 18.0:
            # Long sustained + extreme = likely disengagement
            return 55.0 + min(40.0, (total_dev - 18.0) / 24.0 * 40.0)
        elif sustained_aversion >= 6 and total_dev > 14.0:
            # Moderate sustained
            return 38.0 + min(22.0, sustained_aversion * 2.5)
        elif sustained_aversion >= 4 and total_dev > 20.0:
            # Brief but extreme
            return 28.0 + min(18.0, (total_dev - 20.0) / 15.0 * 18.0)
        elif total_dev > 25.0:
            # Current extreme (but not sustained—might be looking at notes, screen, etc.)
            return 18.0 + min(12.0, (total_dev - 25.0) / 20.0 * 12.0)
        
        # Brief or moderate aversion: likely cognitive processing, NOT resistance
        return 6.0

    def _g3_no_nod(self, buf: list) -> float:
        """
        No-nod: absence of vertical head movement. Research: nodding = agreement;
        absence of nodding = disengagement, resistance, or passive listening.
        
        FALSE POSITIVE REDUCTION: "No nod" is ambiguous—could be neutral listening
        or resistance. Use very strict thresholds and require SUSTAINED stillness.
        Only score high when head is extremely still for extended period.
        """
        if len(buf) < 16:
            return 12.0  # Warmup
        pitches = [b["pitch"] for b in buf[-16:]]
        pitch_std = float(np.std(pitches))
        pitch_range = float(np.max(pitches) - np.min(pitches))
        d = np.diff(pitches)
        zero_crossings = int(np.sum((d[:-1] * d[1:]) < 0))
        
        # Stricter: require VERY still head for EXTENDED period
        # Many people naturally don't nod much; avoid flagging them
        if pitch_std < 0.5 and pitch_range < 1.8 and zero_crossings < 1:
            return 72.0  # Very still = possible resistance (reduced from 82)
        elif pitch_std < 0.8 and pitch_range < 2.5 and zero_crossings < 2:
            return 55.0 + (0.8 - pitch_std) / 0.3 * 12.0
        elif pitch_std < 1.0 and zero_crossings < 1:
            return 45.0
        return 12.0  # Normal (some head movement)

    def _g3_narrowed_pupils(self, cur: dict, buf: list) -> float:
        """
        Narrowed pupils proxy via eye squint (EAR). Research: pupil constriction
        correlates with negative arousal; we proxy via narrowed eyes (lower EAR).
        
        FALSE POSITIVE REDUCTION: Uses baseline comparison (not absolute EAR) and
        requires sustained narrowing. Similar to g2_eye_squint but for resistance context.
        """
        ear = cur.get("ear", 0.2)
        baseline_ear = self._baseline_ear if self._baseline_ear > 0.05 else 0.22
        
        # TEMPORAL: Use median of recent EAR (exclude blinks)
        if len(buf) >= 3:
            recent_ear = [b.get("ear", ear) for b in buf[-3:] if b.get("is_blink", 0) < 0.5]
            if recent_ear:
                ear = float(np.median(recent_ear))
        
        # Compare to baseline: narrowed = EAR below baseline
        ear_ratio = ear / baseline_ear if baseline_ear > 0 else 1.0
        
        # Stricter thresholds: require clear narrowing below baseline
        if ear_ratio < 0.70 and ear < 0.13:  # Strong narrowing
            return 55.0 + min(35.0, (0.70 - ear_ratio) * 140.0)
        if ear_ratio < 0.82 and ear < 0.16:  # Moderate narrowing
            return 38.0 + (0.82 - ear_ratio) * 120.0
        if ear_ratio < 0.90:  # Slight narrowing
            return 22.0 + (0.90 - ear_ratio) * 150.0
        return 15.0  # Normal eyes

    # ----- Group 4 -----
    def _g4_relaxed_exhale(self, buf: list) -> float:
        """
        Relaxed exhale: release of tension. Research: tension release = movement drop
        (stillness after release) + mouth opening (MAR). Uses temporal movement instead
        of spatial face_var (which increases when mouth opens). Fallback: nose_std drop
        or no variance increase when MAR increased.
        """
        if len(buf) < 8:
            return 48.0  # Warmup
        window_before = buf[-8:-4]
        window_now = buf[-3:]
        scales_now = [b.get("face_scale", 50.0) for b in window_now]
        scales_before = [b.get("face_scale", 50.0) for b in window_before]
        avg_scale_now = float(np.mean(scales_now)) if scales_now else 50.0
        avg_scale_before = float(np.mean(scales_before)) if scales_before else 50.0
        var_now_raw = np.mean([b["face_var"] for b in window_now])
        var_before_raw = np.mean([b["face_var"] for b in window_before])
        var_now = var_now_raw / max(avg_scale_now * avg_scale_now, 1e-6)
        var_before = var_before_raw / max(avg_scale_before * avg_scale_before, 1e-6)
        mar_now = float(np.mean([b["mar"] for b in window_now]))
        mar_before = float(np.mean([b["mar"] for b in window_before]))
        recent_mars = [b["mar"] for b in buf[-12:]]
        min_recent_mar = float(np.min(recent_mars)) if recent_mars else mar_before
        baseline_mar = max(self._baseline_mar, 0.04)
        nose_std_now = float(np.mean([b.get("nose_std", 0.0) for b in window_now]))
        nose_std_before = float(np.mean([b.get("nose_std", 0.0) for b in window_before]))

        # 1) Tension drop: prefer movement (stillness after release); fallback when movement missing
        movement_now = float(np.mean([b.get("face_movement", 0.0) for b in window_now]))
        movement_before = float(np.mean([b.get("face_movement", 0.0) for b in window_before]))
        movement_available = movement_before > 1e-9
        if movement_available:
            tension_drop = movement_now < movement_before * 0.78
            strong_tension_drop = movement_before > 0 and movement_now < movement_before * 0.60
        else:
            # Fallback: nose_std drop (>15%) or no variance increase when MAR increased
            mar_increased = mar_before > 0.04 and mar_now > mar_before * 1.03
            nose_drop = nose_std_before > 1e-9 and nose_std_now < nose_std_before * 0.85
            no_var_spike = var_before > 0 and var_now <= var_before * 1.05
            tension_drop = nose_drop or (mar_increased and no_var_spike)
            strong_tension_drop = nose_drop

        # 2) Mouth opening: MAR increase (relaxed 1.03) or above baseline / above recent min
        mouth_opening = (
            (mar_before > 0.04 and mar_now > mar_before * 1.03)
            or (baseline_mar > 0.04 and mar_now > baseline_mar * 1.05)
            or (min_recent_mar > 0.04 and mar_now > min_recent_mar * 1.05)
        )

        # Scoring: both -> 70+; strong tension drop alone -> 58+; moderate tension / mouth with mild tension -> 55-62
        if tension_drop and mouth_opening:
            return 70.0 + min(18.0, (mar_now / max(mar_before, 0.04) - 1.03) * 90.0)
        if strong_tension_drop and mouth_opening:
            return 65.0
        if strong_tension_drop:
            return 58.0 + min(12.0, (1.0 - movement_now / movement_before) * 25.0) if movement_available else 58.0
        if tension_drop:
            return 48.0 + min(10.0, (1.0 - movement_now / movement_before) * 20.0) if movement_available else 52.0
        # Mouth opening with mild tension (no big variance spike) can reach 55-62
        if mouth_opening and var_before > 0 and var_now <= var_before * 1.08:
            return 55.0 + min(7.0, (mar_now / max(baseline_mar, 0.04) - 1.0) * 35.0)
        return 42.0

    def _g4_fixed_gaze(self, buf: list, w: int, h: int) -> float:
        """
        Fixed gaze: stable head orientation = looking at a fixed region (camera or elsewhere).
        Uses yaw/pitch variance (head rotation); low variance = fixated, high = looking around.
        
        FALSE POSITIVE REDUCTION: Use longer window for stability check and require
        sustained fixation (not just momentary stillness). Distinguish genuine focus
        from brief pauses.
        """
        if len(buf) < 12:
            return 50.0  # Warmup
        # Use head orientation (yaw, pitch) with longer window for stability
        window = buf[-14:]
        yaws = [float(b.get("yaw", 0)) for b in window]
        pitches = [float(b.get("pitch", 0)) for b in window]
        std_yaw = float(np.std(yaws))
        std_pitch = float(np.std(pitches))
        head_std = np.sqrt(std_yaw * std_yaw + std_pitch * std_pitch)
        
        # Also check range (not just std) to catch steady drift
        yaw_range = float(np.max(yaws) - np.min(yaws))
        pitch_range = float(np.max(pitches) - np.min(pitches))
        head_range = np.sqrt(yaw_range * yaw_range + pitch_range * pitch_range)
        
        # Stricter: require both low std AND low range
        if head_std < 1.2 and head_range < 3.5:
            return 82.0  # Very fixed (looking at one spot)
        if head_std < 2.5 and head_range < 6.0:
            return 68.0 + (2.5 - head_std) / 1.3 * 10.0  # 68-78
        if head_std < 4.0 and head_range < 10.0:
            return 55.0 + (4.0 - head_std) / 1.5 * 10.0  # 55-65
        if head_std < 6.0:
            return 45.0 + (6.0 - head_std) / 2.0 * 8.0  # 45-53
        return 42.0  # Looking around; low score

    def _g4_smile_transition(self, buf: list, duchenne: float) -> float:
        """
        Smile transition: sustained genuine smile. Research: sustained Duchenne
        + stable mouth (MAR) = authentic positive affect, decision-ready.
        
        FALSE POSITIVE REDUCTION: Require sustained Duchenne score (not just current
        frame) and stable mouth pattern to distinguish genuine smiles from brief
        expressions or noise.
        """
        if len(buf) < 12 or duchenne < 52:
            return 45.0
        mars = [b["mar"] for b in buf[-12:]]
        mar_mean = float(np.mean(mars[-6:]))
        mar_std = float(np.std(mars[-6:]))
        
        # Sustained = stable MAR with slight opening (not just frozen face)
        sustained = mar_mean > 0.14 and mar_std < 0.04
        
        # Also check if Duchenne has been elevated for multiple frames
        # (requires Duchenne to be stored in buffer or estimated)
        if sustained and duchenne >= 62:  # Strong sustained smile
            return 78.0 + min(12.0, (duchenne - 62) * 0.6)
        if sustained and duchenne >= 56:  # Moderate sustained smile
            return 65.0 + min(10.0, (duchenne - 56) * 0.8)
        if duchenne >= 58:  # Current smile but not sustained
            return 55.0 + min(8.0, (duchenne - 58) * 0.5)
        return 46.0
