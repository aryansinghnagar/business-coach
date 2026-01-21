"""
Expression Signifiers Module

Implements 30 B2B-relevant facial expression signifiers from MediaPipe/Azure Face API
landmarks and blendshape-derived features. Each signifier is scored 0-100. A composite
engagement score (0-100) aggregates all 30 for the engagement bar.

Groups:
  1. Interest & Engagement (10): Duchenne, pupil dilation proxy, eyebrow flash, eye contact,
     head tilt, forward lean, mirroring, rhythmic nodding, parted lips, softened forehead.
  2. Cognitive Load (7): look up left/right, lip pucker, eye squint, thinking brow,
     chin stroke (stub), stillness, lowered brow.
  3. Resistance (9): contempt, nose crinkle, lip compression, eye block, jaw clench,
     rapid blink, gaze aversion, no-nod, narrowed pupils (stub), mouth cover (stub).
  4. Decision-Ready (3): relaxed exhale, fixed gaze, social-to-genuine smile.
"""

import time
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple
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
    if len(pts) < 4:
        return 0.2
    x, y = pts[:, 0], pts[:, 1]
    v = np.abs(np.max(y) - np.min(y))
    h = np.max(x) - np.min(x)
    return (v / (h + 1e-6)) if h > 1e-6 else 0.2


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
    from landmarks and optional FaceDetectionResult. Uses a temporal buffer for
    time-based signifiers (blinks, nodding, stillness, etc.).
    """

    def __init__(
        self,
        buffer_frames: int = 45,
        weights_provider: Optional[Callable[[], Dict[str, List[float]]]] = None,
    ):
        self.buffer_frames = max(15, buffer_frames)
        self._buf: deque = deque(maxlen=self.buffer_frames)
        self._weights_provider = weights_provider
        self._landmarks: Optional[np.ndarray] = None
        self._face_result: Optional[FaceDetectionResult] = None
        self._shape: Optional[Tuple[int, int, int]] = None
        # Baseline for dilation proxy and Z
        self._baseline_eye_area: float = 0.0
        self._baseline_z: float = 0.0
        self._blink_start_frames: int = 0
        self._blinks_in_window: int = 0
        self._last_blink_reset: float = 0.0

    def reset(self) -> None:
        """Clear temporal buffer and baselines (e.g. on detection start)."""
        self._buf.clear()
        self._baseline_eye_area = 0.0
        self._baseline_z = 0.0
        self._blink_start_frames = 0
        self._blinks_in_window = 0

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
            mar = mh / mw if mw > 1e-6 else 0.2

        # Head pose from participation-style logic
        nose = lm[NOSE_TIP, :2] if lm.shape[0] > NOSE_TIP else lm[0, :2]
        chin = lm[CHIN, :2] if lm.shape[0] > CHIN else lm[min(16, lm.shape[0] - 1), :2]
        lf = lm[FACE_LEFT, 0] if lm.shape[0] > FACE_LEFT else np.mean(lm[:, 0]) - 50
        rf = lm[FACE_RIGHT, 0] if lm.shape[0] > FACE_RIGHT else np.mean(lm[:, 0]) + 50
        face_cx = (lf + rf) / 2
        face_hw = max(1e-6, abs(rf - lf) / 2)
        head_yaw = np.clip((nose[0] - face_cx) / face_hw, -1.5, 1.5) * 45.0
        dy = chin[1] - nose[1]
        dx = chin[0] - nose[0]
        head_pitch = np.degrees(np.arctan(np.clip(dx / (abs(dy) + 1e-6), -2, 2))) if abs(dy) > 1e-6 else 0.0
        head_pitch = float(abs(head_pitch))
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

        is_blink = 1.0 if ear < 0.16 else 0.0

        # Baselines (EMA)
        if self._baseline_eye_area <= 0 and eye_area > 0:
            self._baseline_eye_area = eye_area
        else:
            self._baseline_eye_area = 0.9 * self._baseline_eye_area + 0.1 * eye_area if eye_area > 0 else self._baseline_eye_area
        if self._baseline_z == 0 and face_z != 0:
            self._baseline_z = face_z
        else:
            self._baseline_z = 0.95 * self._baseline_z + 0.05 * face_z if face_z != 0 else self._baseline_z

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
            "mar": mar, "mouth_w": mw, "mouth_h": mh,
            "face_z": face_z, "gaze_x": gaze_x, "gaze_y": gaze_y,
            "face_var": face_var, "nose_std": nose_std, "is_blink": is_blink,
        }
        self._buf.append(snap)

    def get_all_scores(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        lm = self._landmarks
        fr = self._face_result
        shp = self._shape
        buf = list(self._buf)
        if lm is None or shp is None or len(buf) < 2:
            for k in self._all_keys():
                out[k] = 50.0
            return out

        h, w = shp[0], shp[1]
        cur = buf[-1]
        mouth_pts = _safe(lm, MOUTH)
        ml = lm[MOUTH_LEFT, :2] if lm.shape[0] > MOUTH_LEFT else np.zeros(2)
        mr = lm[MOUTH_RIGHT, :2] if lm.shape[0] > MOUTH_RIGHT else np.zeros(2)

        # --- Group 1: Interest & Engagement ---
        out["g1_duchenne"] = self._g1_duchenne(lm, cur, buf)
        out["g1_pupil_dilation"] = self._g1_pupil_dilation(cur)
        out["g1_eyebrow_flash"] = self._g1_eyebrow_flash(buf)
        out["g1_eye_contact"] = self._g1_eye_contact(cur, w, h)
        out["g1_head_tilt"] = self._g1_head_tilt(cur)
        out["g1_forward_lean"] = self._g1_forward_lean(cur)
        out["g1_mirroring"] = self._g1_mirroring(lm, buf)
        out["g1_rhythmic_nodding"] = self._g1_rhythmic_nodding(buf)
        out["g1_parted_lips"] = self._g1_parted_lips(cur)
        out["g1_softened_forehead"] = self._g1_softened_forehead(lm, cur, buf)

        # --- Group 2: Cognitive Load ---
        out["g2_look_up_lr"] = self._g2_look_up_lr(cur)
        out["g2_lip_pucker"] = self._g2_lip_pucker(lm, cur)
        out["g2_eye_squint"] = self._g2_eye_squint(cur)
        out["g2_thinking_brow"] = self._g2_thinking_brow(lm)
        out["g2_chin_stroke"] = 50.0  # no hand detection
        out["g2_stillness"] = self._g2_stillness(buf)
        out["g2_lowered_brow"] = self._g2_lowered_brow(lm)

        # --- Group 3: Resistance (store as-is; composite uses 100 - x) ---
        out["g3_contempt"] = self._g3_contempt(lm, ml, mr, fr)
        out["g3_nose_crinkle"] = self._g3_nose_crinkle(cur, buf)
        out["g3_lip_compression"] = self._g3_lip_compression(cur)
        out["g3_eye_block"] = self._g3_eye_block(buf)
        out["g3_jaw_clench"] = self._g3_jaw_clench(lm, cur, fr)
        out["g3_rapid_blink"] = self._g3_rapid_blink()
        out["g3_gaze_aversion"] = self._g3_gaze_aversion(cur)
        out["g3_no_nod"] = self._g3_no_nod(buf)
        out["g3_narrowed_pupils"] = self._g3_narrowed_pupils(cur)  # proxy: squint
        out["g3_mouth_cover"] = 50.0  # no hand detection

        # --- Group 4: Decision-Ready ---
        out["g4_relaxed_exhale"] = self._g4_relaxed_exhale(buf)
        out["g4_fixed_gaze"] = self._g4_fixed_gaze(buf, w, h)
        out["g4_smile_transition"] = self._g4_smile_transition(buf, out.get("g1_duchenne", 50))

        for k in out:
            out[k] = max(0.0, min(100.0, float(out[k])))
        return out

    def _all_keys(self) -> List[str]:
        return [
            "g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
            "g1_forward_lean", "g1_mirroring", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead",
            "g2_look_up_lr", "g2_lip_pucker", "g2_eye_squint", "g2_thinking_brow", "g2_chin_stroke",
            "g2_stillness", "g2_lowered_brow",
            "g3_contempt", "g3_nose_crinkle", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
            "g3_rapid_blink", "g3_gaze_aversion", "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover",
            "g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition",
        ]

    def get_composite_score(self, scores: Optional[Dict[str, float]] = None) -> float:
        if scores is None:
            scores = self.get_all_scores()
        keys = self._all_keys()
        W = self._weights_provider() if self._weights_provider else {"signifier": [1.0] * 30, "group": [0.35, 0.15, 0.35, 0.15]}
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
            return float(np.mean([scores[k] for k in grp_keys if k in scores])) if grp_keys else 50.0

        g1k = ["g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
               "g1_forward_lean", "g1_mirroring", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead"]
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

        composite = gw[0] * g1 + gw[1] * g2 + gw[2] * g3 + gw[3] * g4
        return float(max(0.0, min(100.0, composite)))

    # ----- Group 1 -----
    def _g1_duchenne(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        # Smile (mouth corners up) + slight eye squinch (EAR dip) = Duchenne proxy
        mar = cur.get("mar", 0.2)
        ear = cur.get("ear", 0.2)
        mouth_pts = _safe(lm, MOUTH)
        if len(mouth_pts) < 8:
            return 50.0
        ly = float(mouth_pts[0, 1])
        ry = float(mouth_pts[5, 1]) if mouth_pts.shape[0] > 5 else ly
        mid_y = np.mean(mouth_pts[:, 1])
        # Corners above mid = smile
        smile = (mid_y - (ly + ry) / 2) > 2.0
        squinch = 0.18 < ear < 0.28  # slight narrowing
        if smile and squinch:
            return 85.0
        if smile:
            return 65.0
        if squinch:
            return 45.0
        return 40.0

    def _g1_pupil_dilation(self, cur: dict) -> float:
        # Proxy: eye area vs baseline (wider = higher)
        area = cur.get("eye_area", 0.0)
        base = max(1e-6, self._baseline_eye_area)
        r = area / base
        if r >= 1.05:
            return 50.0 + min(50.0, (r - 1.0) * 200.0)
        if r >= 0.95:
            return 50.0
        return max(0.0, 50.0 + (r - 1.0) * 100.0)

    def _g1_eyebrow_flash(self, buf: list) -> float:
        if len(buf) < 10:
            return 50.0
        heights = [(b["eyebrow_l"] + b["eyebrow_r"]) / 2 for b in buf]
        baseline = np.mean(heights[: max(1, len(heights) // 2)])
        cur = heights[-1]
        if cur > baseline + 2.0:  # raised
            return 75.0
        return 50.0

    def _g1_eye_contact(self, cur: dict, w: int, h: int) -> float:
        gx, gy = cur.get("gaze_x", 0), cur.get("gaze_y", 0)
        d = np.sqrt(gx * gx + gy * gy)
        md = np.sqrt(w * w + h * h) / 2
        nd = min(1.0, d / max(md, 1e-6))
        yaw = abs(cur.get("yaw", 0))
        c = (1.0 - nd) * 100.0
        c *= (1.0 - min(0.5, yaw / 45.0))
        return float(c)

    def _g1_head_tilt(self, cur: dict) -> float:
        roll = abs(cur.get("roll", 0))
        if 5 <= roll <= 18:
            return 60.0 + min(35.0, (roll - 5) * 2.5)
        if 0 <= roll < 5:
            return 40.0 + roll * 4.0
        if 18 < roll <= 35:
            return 90.0 - (roll - 18) * 1.5
        return max(20.0, 50.0 - roll * 0.5)

    def _g1_forward_lean(self, cur: dict) -> float:
        z = cur.get("face_z", 0)
        base = self._baseline_z
        if base == 0 or z == 0:
            return 50.0
        if z < base * 0.97:
            return 50.0 + min(50.0, (1.0 - z / base) * 150.0)
        return 50.0

    def _g1_mirroring(self, lm: np.ndarray, buf: list) -> float:
        # Symmetry as proxy for mirroring (symmetric pose)
        if len(buf) < 2 or lm.shape[0] < 20:
            return 55.0
        n2 = lm.shape[0] // 2
        left = lm[:n2, :2]
        right = lm[n2 : 2 * n2, :2] if lm.shape[0] >= 2 * n2 else lm[n2:, :2]
        cx = np.mean(lm[:, 0])
        right_m = 2 * cx - right[:, 0]
        err = np.mean(np.abs(left[:, 0] - right_m[: left.shape[0]])) if left.shape[0] == right.shape[0] else 30.0
        sym = 100.0 - min(100.0, err)
        return float(sym)

    def _g1_rhythmic_nodding(self, buf: list) -> float:
        if len(buf) < 15:
            return 40.0
        pitches = [b["pitch"] for b in buf[-20:]]
        if np.std(pitches) < 1.0:
            return 35.0
        # Simple: 1–2 “oscillations” in pitch
        d = np.diff(pitches)
        crosses = np.sum((d[:-1] * d[1:]) < 0)
        if 1 <= crosses <= 4:
            return 70.0
        return 45.0

    def _g1_parted_lips(self, cur: dict) -> float:
        mar = cur.get("mar", 0.2)
        if 0.12 <= mar <= 0.35:
            return 50.0 + (mar - 0.12) / 0.23 * 45.0
        if 0.08 <= mar < 0.12:
            return 35.0 + (mar - 0.08) / 0.04 * 15.0
        if 0.35 < mar <= 0.5:
            return 85.0 - (mar - 0.35) / 0.15 * 20.0
        return 40.0

    def _g1_softened_forehead(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        # Low brow tension: eyebrow variance low, not furrowed
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        if len(lb) < 2 or len(rb) < 2:
            return 55.0
        v = float(np.var(np.vstack([lb[:, :2], rb[:, :2]])))
        if v < 5.0:
            return 75.0
        if v < 20.0:
            return 65.0
        return max(35.0, 70.0 - v * 0.5)

    # ----- Group 2 -----
    def _g2_look_up_lr(self, cur: dict) -> float:
        p, y = cur.get("pitch", 0), cur.get("yaw", 0)
        # Up = negative pitch (nose up); left/right = yaw
        if -25 <= p <= -5 and 10 <= abs(y) <= 40:
            return 70.0
        if -20 <= p <= 0 and 5 <= abs(y) <= 35:
            return 55.0
        return 45.0

    def _g2_lip_pucker(self, lm: np.ndarray, cur: dict) -> float:
        mar = cur.get("mar", 0.2)
        mw = cur.get("mouth_w", 40.0)
        # Pucker: MAR elevated relative to typical for that width, or width decreased
        if mar > 0.35 and mw < 50:
            return 70.0
        if mar > 0.3:
            return 55.0
        return 40.0

    def _g2_eye_squint(self, cur: dict) -> float:
        ear = cur.get("ear", 0.2)
        if 0.10 <= ear <= 0.18:
            return 50.0 + (0.18 - ear) / 0.08 * 45.0
        if ear < 0.10:
            return 85.0
        return 35.0

    def _g2_thinking_brow(self, lm: np.ndarray) -> float:
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        if len(lb) < 2 or len(rb) < 2:
            return 50.0
        ly, ry = np.mean(lb[:, 1]), np.mean(rb[:, 1])
        d = abs(ly - ry)
        if 2 <= d <= 12:
            return 70.0
        if 1 <= d <= 15:
            return 55.0
        return 45.0

    def _g2_stillness(self, buf: list) -> float:
        if len(buf) < 10:
            return 50.0
        vars = [b["face_var"] for b in buf[-12:]]
        m = np.mean(vars)
        if m < 5.0:
            return 75.0
        if m < 15.0:
            return 60.0
        return 45.0

    def _g2_lowered_brow(self, lm: np.ndarray) -> float:
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        le = _safe(lm, LEFT_EYE)
        re = _safe(lm, RIGHT_EYE)
        if len(lb) < 2 or len(le) < 2:
            return 50.0
        brow_y = (np.mean(lb[:, 1]) + np.mean(rb[:, 1])) / 2
        eye_y = (np.mean(le[:, 1]) + np.mean(re[:, 1])) / 2
        dist = eye_y - brow_y  # smaller = furrow
        if dist < 8:
            return 75.0
        if dist < 12:
            return 60.0
        return 45.0

    # ----- Group 3 -----
    def _g3_contempt(self, lm: np.ndarray, ml: np.ndarray, mr: np.ndarray, fr: Optional[FaceDetectionResult]) -> float:
        if fr and getattr(fr, "emotions", None) and isinstance(fr.emotions, dict):
            c = fr.emotions.get("contempt", 0.0) or 0.0
            return min(100.0, c * 150.0)
        if lm.shape[0] > max(MOUTH_LEFT, MOUTH_RIGHT):
            ly, ry = float(ml[1]), float(mr[1])
            if abs(ly - ry) > 4:
                return min(100.0, 30.0 + abs(ly - ry) * 5.0)
        return 20.0

    def _g3_nose_crinkle(self, cur: dict, buf: list) -> float:
        ns = cur.get("nose_std", 0)
        if len(buf) >= 5:
            prev = np.mean([b.get("nose_std", 0) for b in buf[-6:-1]])
            if ns > prev * 1.3 and ns > 3:
                return 65.0
        return 15.0

    def _g3_lip_compression(self, cur: dict) -> float:
        mar = cur.get("mar", 0.2)
        if mar < 0.06:
            return 80.0
        if mar < 0.10:
            return 50.0 + (0.10 - mar) / 0.04 * 30.0
        return 25.0

    def _g3_eye_block(self, buf: list) -> float:
        # Long closure: consecutive frames with EAR < 0.12
        if len(buf) < 5:
            return 10.0
        run = 0
        for b in reversed(buf):
            if b.get("ear", 0.2) < 0.12:
                run += 1
            else:
                break
        # ~30fps: 15 frames ≈ 0.5 s
        if run >= 15:
            return 90.0
        if run >= 8:
            return 50.0 + (run - 8) / 7 * 40.0
        return 10.0

    def _g3_jaw_clench(self, lm: np.ndarray, cur: dict, fr: Optional[FaceDetectionResult]) -> float:
        mar = cur.get("mar", 0.2)
        mouth_pts = _safe(lm, MOUTH)
        if len(mouth_pts) < 8:
            return 20.0
        # Corners down + tight MAR
        ly = float(mouth_pts[0, 1])
        ry = float(mouth_pts[5, 1]) if mouth_pts.shape[0] > 5 else ly
        mid = np.mean(mouth_pts[:, 1])
        corners_down = (ly + ry) / 2 > mid + 2
        if mar < 0.08 and corners_down:
            return 75.0
        if mar < 0.10:
            return 40.0
        return 15.0

    def _g3_rapid_blink(self) -> float:
        # Blinks in last 2s
        b = self._blinks_in_window
        if b >= 5:
            return 85.0
        if b >= 3:
            return 50.0 + (b - 3) * 15.0
        return 10.0

    def _g3_gaze_aversion(self, cur: dict) -> float:
        p = cur.get("pitch", 0)
        if p > 15:
            return 40.0 + min(50.0, (p - 15) * 2.0)
        if p > 8:
            return 25.0 + (p - 8) * 2.0
        return 15.0

    def _g3_no_nod(self, buf: list) -> float:
        if len(buf) < 12:
            return 10.0
        yaws = [b["yaw"] for b in buf[-15:]]
        d = np.diff(yaws)
        crosses = np.sum((d[:-1] * d[1:]) < 0)
        if 2 <= crosses <= 6 and np.std(yaws) > 1.5:
            return 70.0
        return 15.0

    def _g3_narrowed_pupils(self, cur: dict) -> float:
        ear = cur.get("ear", 0.2)
        if ear < 0.14:
            return 50.0 + (0.14 - ear) / 0.14 * 40.0
        return 20.0

    # ----- Group 4 -----
    def _g4_relaxed_exhale(self, buf: list) -> float:
        if len(buf) < 10:
            return 50.0
        var_now = np.mean([b["face_var"] for b in buf[-3:]])
        var_before = np.mean([b["face_var"] for b in buf[-10:-5]])
        mar_now = np.mean([b["mar"] for b in buf[-3:]])
        mar_before = np.mean([b["mar"] for b in buf[-10:-5]])
        if var_now < var_before * 0.7 and mar_now > mar_before * 1.05:
            return 75.0
        if var_now < var_before * 0.85:
            return 60.0
        return 45.0

    def _g4_fixed_gaze(self, buf: list, w: int, h: int) -> float:
        if len(buf) < 8:
            return 50.0
        gx = [b["gaze_x"] for b in buf[-10:]]
        gy = [b["gaze_y"] for b in buf[-10:]]
        std = np.sqrt(np.var(gx) + np.var(gy))
        if std < 15:
            return 70.0
        if std < 35:
            return 55.0
        return 40.0

    def _g4_smile_transition(self, buf: list, duchenne: float) -> float:
        if len(buf) < 20 or duchenne < 50:
            return 45.0
        mars = [b["mar"] for b in buf[-20:]]
        # Sustained moderate MAR + Duchenne
        sustained = np.mean(mars[-10:]) > 0.18 and np.std(mars[-10:]) < 0.04
        if sustained and duchenne >= 60:
            return 80.0
        if duchenne >= 55:
            return 60.0
        return 50.0
