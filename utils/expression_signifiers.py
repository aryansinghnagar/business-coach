"""
Expression Signifiers Module

Implements 30 B2B-relevant facial expression signifiers from MediaPipe/Azure Face API
landmarks and blendshape-derived features. Each signifier is scored 0-100. A composite
engagement score (0-100) aggregates all 30 for the engagement bar.

Groups:
  1. Interest & Engagement (10): Duchenne, pupil dilation proxy, eyebrow flash, eye contact,
     head tilt, forward lean, facial symmetry, rhythmic nodding, parted lips, softened forehead.
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
        self._baseline_ear: float = 0.0  # For Duchenne squinch detection
        self._pupil_dilation_history: deque = deque(maxlen=5)  # Smooth pupil dilation score
        self._last_pupil_dilation_score: float = 50.0  # Last valid score (for blink frames)
        self._blink_start_frames: int = 0
        self._blinks_in_window: int = 0
        self._last_blink_reset: float = 0.0

    def reset(self) -> None:
        """Clear temporal buffer and baselines (e.g. on detection start)."""
        self._buf.clear()
        self._baseline_eye_area = 0.0
        self._baseline_z = 0.0
        self._baseline_ear = 0.0
        self._pupil_dilation_history.clear()
        self._last_pupil_dilation_score = 50.0
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

        # Baselines (EMA) - only update when not blinking to avoid contamination
        if is_blink < 0.5:  # Only update baseline when not blinking
            if self._baseline_eye_area <= 0 and eye_area > 0:
                self._baseline_eye_area = eye_area
            elif eye_area > 0:
                self._baseline_eye_area = 0.95 * self._baseline_eye_area + 0.05 * eye_area
        if self._baseline_z == 0 and face_z != 0:
            self._baseline_z = face_z
        else:
            self._baseline_z = 0.95 * self._baseline_z + 0.05 * face_z if face_z != 0 else self._baseline_z
        # Baseline EAR for Duchenne squinch detection (exclude blinks)
        if is_blink < 0.5:  # Only update baseline when not blinking
            if self._baseline_ear <= 0 and ear > 0:
                self._baseline_ear = ear
            elif ear > 0:
                self._baseline_ear = 0.95 * self._baseline_ear + 0.05 * ear

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
            "face_scale": face_scale,  # For size/position invariance
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
        out["g1_facial_symmetry"] = self._g1_facial_symmetry(lm, buf)
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
            "g1_forward_lean", "g1_facial_symmetry", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead",
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

        composite = gw[0] * g1 + gw[1] * g2 + gw[2] * g3 + gw[3] * g4
        return float(max(0.0, min(100.0, composite)))

    # ----- Group 1 -----
    def _g1_duchenne(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        # Duchenne smile: genuine smile with eye squinch (crow's feet)
        # Smile = mouth corners raised; Squinch = EAR reduction relative to baseline
        ear = cur.get("ear", 0.2)
        face_scale = cur.get("face_scale", 50.0)
        mouth_pts = _safe(lm, MOUTH)
        if len(mouth_pts) < 8:
            return 50.0
        
        # Calculate smile intensity (gradual, not binary)
        ly = float(mouth_pts[0, 1])
        ry = float(mouth_pts[5, 1]) if mouth_pts.shape[0] > 5 else ly
        mid_y = float(np.mean(mouth_pts[:, 1]))
        corner_lift = (mid_y - (ly + ry) / 2)  # Positive = corners up
        smile_threshold = face_scale * 0.03  # Minimum for smile
        smile_intensity = min(1.0, max(0.0, corner_lift / max(face_scale * 0.12, 1e-6)))  # 0-1 scale
        
        # Calculate squinch: EAR reduction relative to baseline (more accurate)
        baseline_ear = self._baseline_ear if self._baseline_ear > 0 else 0.25
        if baseline_ear <= 0:
            return 50.0
        ear_ratio = ear / baseline_ear
        # Squinch: EAR 85-95% of baseline (slight narrowing, not full squint)
        squinch_intensity = 0.0
        if 0.85 <= ear_ratio <= 0.95:
            squinch_intensity = 1.0 - abs(ear_ratio - 0.90) / 0.10  # Peak at 0.90
        elif ear_ratio < 0.85:
            squinch_intensity = 0.5  # Too much squint, less genuine
        
        # Combine: Duchenne = smile + squinch
        if corner_lift > smile_threshold and squinch_intensity > 0.3:
            # Both present: genuine Duchenne
            combined = 0.6 * smile_intensity + 0.4 * squinch_intensity
            return 50.0 + combined * 50.0  # 50-100
        elif corner_lift > smile_threshold:
            # Smile only: social smile
            return 50.0 + smile_intensity * 30.0  # 50-80
        elif squinch_intensity > 0.3:
            # Squinch only: neutral
            return 45.0 + squinch_intensity * 10.0  # 45-55
        return 30.0  # No smile, no squinch

    def _g1_pupil_dilation(self, cur: dict) -> float:
        # Proxy: eye area vs baseline (wider = higher)
        # Exclude blinks and use temporal smoothing for stability
        is_blink = cur.get("is_blink", 0.0)
        area = cur.get("eye_area", 0.0)
        base = max(1e-6, self._baseline_eye_area)
        
        # During blinks, return last valid score (don't use blink data)
        if is_blink > 0.5:
            # Use last valid score with slight decay toward neutral
            smoothed = 0.9 * self._last_pupil_dilation_score + 0.1 * 50.0
            self._last_pupil_dilation_score = smoothed
            return float(smoothed)
        
        # Calculate ratio only when not blinking
        r = area / base if base > 0 else 1.0
        
        # Score calculation: 0-100 range
        # r < 0.9: eyes narrowed (low dilation) -> 0-40
        # r 0.9-1.0: normal -> 40-50
        # r 1.0-1.1: slightly wider -> 50-60
        # r > 1.1: dilated -> 60-100
        if r >= 1.1:
            score = 60.0 + min(40.0, (r - 1.1) * 200.0)  # 60-100 for r 1.1-1.3+
        elif r >= 1.05:
            score = 55.0 + (r - 1.05) / 0.05 * 5.0  # 55-60 for r 1.05-1.1
        elif r >= 1.0:
            score = 50.0 + (r - 1.0) / 0.05 * 5.0  # 50-55 for r 1.0-1.05
        elif r >= 0.95:
            score = 45.0 + (r - 0.95) / 0.05 * 5.0  # 45-50 for r 0.95-1.0
        elif r >= 0.9:
            score = 40.0 + (r - 0.9) / 0.05 * 5.0  # 40-45 for r 0.9-0.95
        else:
            score = max(0.0, 40.0 + (r - 0.9) * 400.0)  # 0-40 for r < 0.9
        
        # Temporal smoothing: blend with recent history to reduce fluctuation
        self._pupil_dilation_history.append(score)
        if len(self._pupil_dilation_history) >= 3:
            # Weighted average: 70% current, 30% recent average
            recent_avg = float(np.mean(list(self._pupil_dilation_history)[:-1]))
            smoothed_score = 0.7 * score + 0.3 * recent_avg
        else:
            smoothed_score = score
        
        # Update last valid score
        self._last_pupil_dilation_score = smoothed_score
        
        return float(max(0.0, min(100.0, smoothed_score)))

    def _g1_eyebrow_flash(self, buf: list) -> float:
        # Eyebrow flash: rapid raise and return (surprise/acknowledgment)
        if len(buf) < 15:
            return 50.0
        heights = [(b["eyebrow_l"] + b["eyebrow_r"]) / 2 for b in buf]
        face_scale = buf[-1].get("face_scale", 50.0)
        threshold = face_scale * 0.04
        
        # Baseline from first half
        baseline = float(np.mean(heights[:max(1, len(heights) // 2)]))
        cur = heights[-1]
        
        # Check for recent raise (last 5 frames)
        recent_heights = heights[-5:]
        max_recent = float(np.max(recent_heights))
        min_recent = float(np.min(recent_heights))
        
        # Flash pattern: raised recently, now returning
        raised = max_recent > baseline + threshold
        returning = cur < max_recent - threshold * 0.5  # Dropped from peak
        
        if raised and returning:
            # Strong flash: rapid raise and return
            flash_magnitude = (max_recent - baseline) / max(face_scale * 0.15, 1e-6)
            return 60.0 + min(35.0, flash_magnitude * 100.0)  # 60-95
        elif raised:
            # Raised but not yet returning
            raise_magnitude = (cur - baseline) / max(face_scale * 0.15, 1e-6)
            return 50.0 + min(20.0, raise_magnitude * 50.0)  # 50-70
        return 50.0

    def _g1_eye_contact(self, cur: dict, w: int, h: int) -> float:
        # Eye contact: gaze direction + head orientation both toward camera
        gx, gy = cur.get("gaze_x", 0), cur.get("gaze_y", 0)
        yaw = cur.get("yaw", 0)
        pitch = cur.get("pitch", 0)
        face_scale = cur.get("face_scale", 50.0)
        
        # Gaze distance from center (normalized by face scale)
        gaze_dist = np.sqrt(gx * gx + gy * gy)
        gaze_normalized = min(1.0, gaze_dist / max(face_scale * 0.4, 1e-6))  # 40% of inter-ocular = good threshold
        
        # Head orientation penalty (yaw and pitch)
        yaw_penalty = min(1.0, abs(yaw) / 30.0)  # 30° = significant turn
        pitch_penalty = min(1.0, abs(pitch) / 20.0)  # 20° = looking up/down
        
        # Combine: gaze quality (0-100) penalized by head orientation
        gaze_score = (1.0 - gaze_normalized) * 100.0
        head_penalty = (yaw_penalty + pitch_penalty) / 2.0
        final_score = gaze_score * (1.0 - head_penalty * 0.6)  # Head orientation can reduce by up to 60%
        
        return float(max(0.0, min(100.0, final_score)))

    def _g1_head_tilt(self, cur: dict) -> float:
        # Head tilt: slight tilt (5-15°) shows interest, extreme tilt shows disengagement
        roll = abs(cur.get("roll", 0))
        
        # Optimal: 5-15° (shows interest, listening)
        if 5 <= roll <= 15:
            # Peak engagement at ~10°
            optimal_dist = abs(roll - 10.0)
            return 70.0 + (1.0 - optimal_dist / 5.0) * 25.0  # 70-95
        
        # Very slight tilt (0-5°): neutral-positive
        if 0 <= roll < 5:
            return 55.0 + roll * 3.0  # 55-70
        
        # Moderate tilt (15-25°): still engaged but less so
        if 15 < roll <= 25:
            return 60.0 - (roll - 15) * 2.0  # 60-40
        
        # Extreme tilt (>25°): disengaged
        return max(10.0, 40.0 - (roll - 25) * 1.5)  # 40-10

    def _g1_forward_lean(self, cur: dict) -> float:
        z = cur.get("face_z", 0)
        base = self._baseline_z
        if base == 0 or z == 0:
            return 50.0
        if z < base * 0.97:
            return 50.0 + min(50.0, (1.0 - z / base) * 150.0)
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
        
        if total_error < 0.03:  # < 3% error: excellent symmetry
            # Map 0-3% to 100-90
            symmetry_score = 100.0 - (total_error / 0.03) * 10.0
        elif total_error < 0.06:  # 3-6% error: good symmetry
            # Map 3-6% to 90-70
            symmetry_score = 90.0 - ((total_error - 0.03) / 0.03) * 20.0
        elif total_error < 0.10:  # 6-10% error: moderate symmetry
            # Map 6-10% to 70-50
            symmetry_score = 70.0 - ((total_error - 0.06) / 0.04) * 20.0
        elif total_error < 0.15:  # 10-15% error: poor symmetry
            # Map 10-15% to 50-30
            symmetry_score = 50.0 - ((total_error - 0.10) / 0.05) * 20.0
        else:  # > 15% error: very poor symmetry
            # Map 15-30% to 30-0 (cap at 30% error for scoring)
            capped_error = min(total_error, 0.30)
            symmetry_score = max(0.0, 30.0 - ((capped_error - 0.15) / 0.15) * 30.0)
        
        return float(max(0.0, min(100.0, symmetry_score)))

    def _g1_rhythmic_nodding(self, buf: list) -> float:
        # Rhythmic nodding: vertical head movement (pitch) showing agreement
        if len(buf) < 20:
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
        # Normalize variance by face scale squared (variance scales with area)
        face_scale = cur.get("face_scale", 50.0)
        v_normalized = v / max(face_scale * face_scale, 1e-6)
        # Thresholds: 5.0 -> 0.002, 20.0 -> 0.008 (relative to face scale^2)
        if v_normalized < 0.002:
            return 75.0
        if v_normalized < 0.008:
            return 65.0
        return max(35.0, 70.0 - v_normalized * 5000.0)

    # ----- Group 2 -----
    def _g2_look_up_lr(self, cur: dict) -> float:
        # Looking up and left/right: cognitive processing (accessing memory)
        p, y = cur.get("pitch", 0), cur.get("yaw", 0)
        
        # Looking up: positive pitch (nose up, eyes up) - matches Azure convention
        # Looking left/right: yaw
        looking_up = p > 5  # Positive pitch = looking up
        looking_lr = abs(y) > 10  # Significant left/right turn
        
        if looking_up and looking_lr:
            # Strong cognitive load: looking up AND to side
            up_intensity = min(1.0, p / 25.0)  # 0-1, peaks at 25° (p is positive when looking up)
            lr_intensity = min(1.0, abs(y) / 40.0)  # 0-1, peaks at 40°
            combined = (up_intensity + lr_intensity) / 2.0
            return 50.0 + combined * 40.0  # 50-90
        elif looking_up:
            # Just looking up
            up_intensity = min(1.0, p / 20.0)  # p is positive when looking up
            return 50.0 + up_intensity * 25.0  # 50-75
        elif looking_lr:
            # Just looking left/right (less cognitive load indicator)
            return 50.0
        return 45.0  # Looking forward

    def _g2_lip_pucker(self, lm: np.ndarray, cur: dict) -> float:
        # Lip pucker: cognitive processing (thinking, evaluating)
        mar = cur.get("mar", 0.2)
        face_scale = cur.get("face_scale", 50.0)
        mw = cur.get("mouth_w", 40.0)
        
        # Pucker: high MAR (vertical stretch) relative to mouth width
        # Normalize mouth width by face scale
        mw_normalized = mw / max(face_scale, 1e-6)
        
        # Pucker indicators:
        # 1. High MAR (>0.3) with relatively narrow mouth
        # 2. MAR significantly elevated for the mouth size
        if mar > 0.35 and mw_normalized < 0.15:  # Very high MAR, narrow mouth
            return 75.0
        elif mar > 0.30 and mw_normalized < 0.20:
            return 60.0 + (mar - 0.30) / 0.05 * 10.0  # 60-70
        elif mar > 0.25:
            return 50.0 + (mar - 0.25) / 0.10 * 15.0  # 50-65
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
        # Get face scale from buffer if available, else estimate from landmarks
        if hasattr(self, '_buf') and len(self._buf) > 0:
            face_scale = list(self._buf)[-1].get("face_scale", 50.0)
        else:
            # Estimate from face width
            face_scale = float(np.max(lm[:, 0]) - np.min(lm[:, 0])) * 0.5 if lm.shape[0] > 0 else 50.0
        face_scale = max(20.0, face_scale)
        # Thresholds relative to face scale: 2px -> 4%, 12px -> 24%, 15px -> 30%
        d_rel = d / face_scale
        if 0.04 <= d_rel <= 0.24:
            return 70.0
        if 0.02 <= d_rel <= 0.30:
            return 55.0
        return 45.0

    def _g2_stillness(self, buf: list) -> float:
        if len(buf) < 10:
            return 50.0
        vars_list = [b["face_var"] for b in buf[-12:]]
        scales = [b.get("face_scale", 50.0) for b in buf[-12:]]
        # Normalize variance by average face scale squared
        avg_scale = float(np.mean(scales)) if scales else 50.0
        avg_scale_sq = max(400.0, avg_scale * avg_scale)  # min 20^2
        m = float(np.mean(vars_list))
        m_normalized = m / avg_scale_sq
        # Thresholds: 5.0 -> 0.002, 15.0 -> 0.006 (relative to face scale^2)
        if m_normalized < 0.002:
            return 75.0
        if m_normalized < 0.006:
            return 60.0
        return 45.0

    def _g2_lowered_brow(self, lm: np.ndarray) -> float:
        # Lowered brow (furrowed): cognitive load or concentration
        # NOTE: This is Group 2 (Cognitive Load), so HIGH score = high cognitive load
        # But in Group 3 context, furrowed brow = resistance = HIGH score
        # This function is in Group 2, so high score = cognitive load (good for engagement)
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        le = _safe(lm, LEFT_EYE)
        re = _safe(lm, RIGHT_EYE)
        if len(lb) < 2 or len(le) < 2:
            return 50.0
        
        brow_y = (np.mean(lb[:, 1]) + np.mean(rb[:, 1])) / 2
        eye_y = (np.mean(le[:, 1]) + np.mean(re[:, 1])) / 2
        dist = eye_y - brow_y  # Smaller distance = furrowed (brow lowered)
        
        # Get face scale
        if hasattr(self, '_buf') and len(self._buf) > 0:
            face_scale = list(self._buf)[-1].get("face_scale", 50.0)
        else:
            face_scale = float(np.max(lm[:, 0]) - np.min(lm[:, 0])) * 0.5 if lm.shape[0] > 0 else 50.0
        face_scale = max(20.0, face_scale)
        
        # Normalize distance
        dist_rel = dist / face_scale
        
        # Furrowed: dist_rel < 0.20 (brow close to eye)
        # Normal: dist_rel 0.20-0.30
        # Raised: dist_rel > 0.30
        if dist_rel < 0.15:  # Strongly furrowed
            return 80.0
        elif dist_rel < 0.20:  # Moderately furrowed
            return 65.0 + (0.20 - dist_rel) / 0.05 * 10.0  # 65-75
        elif dist_rel < 0.25:  # Slightly furrowed
            return 55.0 + (0.25 - dist_rel) / 0.05 * 10.0  # 55-65
        return 45.0  # Normal or raised

    # ----- Group 3 -----
    def _g3_contempt(self, lm: np.ndarray, ml: np.ndarray, mr: np.ndarray, fr: Optional[FaceDetectionResult]) -> float:
        if fr and getattr(fr, "emotions", None) and isinstance(fr.emotions, dict):
            c = fr.emotions.get("contempt", 0.0) or 0.0
            return min(100.0, c * 150.0)
        if lm.shape[0] > max(MOUTH_LEFT, MOUTH_RIGHT):
            ly, ry = float(ml[1]), float(mr[1])
            # Get face scale from buffer if available
            if hasattr(self, '_buf') and len(self._buf) > 0:
                face_scale = list(self._buf)[-1].get("face_scale", 50.0)
            else:
                face_scale = float(np.max(lm[:, 0]) - np.min(lm[:, 0])) * 0.5 if lm.shape[0] > 0 else 50.0
            face_scale = max(20.0, face_scale)
            # Threshold relative to face scale: 4px -> 8%
            asymmetry = abs(ly - ry)
            if asymmetry > face_scale * 0.08:
                return min(100.0, 30.0 + (asymmetry / face_scale) * 625.0)  # Scale score by relative asymmetry
        return 20.0

    def _g3_nose_crinkle(self, cur: dict, buf: list) -> float:
        # Nose crinkle shortens vertical extent (bridge→tip); detect drop vs recent average.
        nh = cur.get("nose_height", 0.0)
        if nh <= 0 or len(buf) < 6:
            return 12.0
        recent = [b.get("nose_height", nh) for b in buf[-8:-1] if b.get("nose_height", 0) > 0]
        if not recent:
            return 12.0
        avg = float(np.mean(recent))
        if avg <= 1e-6:
            return 12.0
        if nh < avg * 0.88:  # clear shortening
            return 55.0 + min(35.0, (1.0 - nh / avg) * 80.0)
        if nh < avg * 0.95:
            return 35.0
        return 12.0

    def _g3_lip_compression(self, cur: dict) -> float:
        # Use inner MAR (d13–14 / d61–17) when available; more accurate for pressed lips.
        mar = cur.get("mar_inner", cur.get("mar", 0.2))
        if mar < 0.05:
            return 88.0
        if mar < 0.08:
            return 68.0 + (0.08 - mar) / 0.03 * 10.0
        if mar < 0.12:
            return 40.0 + (0.12 - mar) / 0.04 * 20.0
        if mar < 0.18:
            return 20.0 + (0.18 - mar) / 0.06 * 15.0
        return 15.0

    def _g3_eye_block(self, buf: list) -> float:
        # Prolonged closure (blocking): EAR < 0.10 to exclude normal blink tails.
        # ~30fps: 10 frames ≈ 0.33 s, 20 frames ≈ 0.67 s.
        if len(buf) < 5:
            return 8.0
        run = 0
        for b in reversed(buf):
            if b.get("ear", 0.2) < 0.10:
                run += 1
            else:
                break
        if run >= 20:
            return 90.0
        if run >= 10:
            return 50.0 + (run - 10) / 10 * 38.0
        return 8.0

    def _g3_jaw_clench(self, lm: np.ndarray, cur: dict, fr: Optional[FaceDetectionResult]) -> float:
        # MAR-primary: tight lips/jaw; corners-down is a supporting cue (frown).
        mar = cur.get("mar_inner", cur.get("mar", 0.2))
        face_scale = cur.get("face_scale", 50.0)
        mouth_pts = _safe(lm, MOUTH)
        corners_down = False
        if len(mouth_pts) >= 6:
            ly = float(mouth_pts[0, 1])   # 61 left
            ry = float(mouth_pts[5, 1])   # 17 right
            mid = float(np.mean(mouth_pts[:, 1]))
            # Threshold relative to face scale: 3px -> 6%
            threshold = face_scale * 0.06
            corners_down = (ly + ry) / 2 > mid + threshold
        if mar < 0.07:
            return min(95.0, 78.0 + (12.0 if corners_down else 0.0))
        if mar < 0.10:
            return 52.0 + (0.10 - mar) / 0.03 * 18.0 + (8.0 if corners_down else 0.0)
        if mar < 0.13:
            return 28.0 + (0.13 - mar) / 0.03 * 18.0 + (7.0 if corners_down else 0.0)
        return 12.0

    def _g3_rapid_blink(self) -> float:
        # Blinks in last 2s
        b = self._blinks_in_window
        if b >= 5:
            return 85.0
        if b >= 3:
            return 50.0 + (b - 3) * 15.0
        return 10.0

    def _g3_gaze_aversion(self, cur: dict) -> float:
        # Gaze aversion: looking away (resistance/disengagement)
        # High score = high aversion = resistance
        p = cur.get("pitch", 0)
        y = cur.get("yaw", 0)
        
        # Combine pitch and yaw for total gaze deviation
        pitch_dev = abs(p)
        yaw_dev = abs(y)
        total_dev = np.sqrt(pitch_dev * pitch_dev + yaw_dev * yaw_dev)
        
        # Significant aversion: >20° total deviation
        if total_dev > 20.0:
            return 30.0 + min(60.0, (total_dev - 20.0) / 30.0 * 60.0)  # 30-90
        elif total_dev > 10.0:
            return 20.0 + (total_dev - 10.0) / 10.0 * 10.0  # 20-30
        elif total_dev > 5.0:
            return 15.0 + (total_dev - 5.0) / 5.0 * 5.0  # 15-20
        return 10.0  # Looking forward

    def _g3_no_nod(self, buf: list) -> float:
        # No nod: absence of nodding (resistance indicator)
        # High score = no nodding = resistance
        if len(buf) < 20:
            return 10.0
        
        # Check for nodding in pitch (vertical movement)
        pitches = [b["pitch"] for b in buf[-30:]]
        pitch_std = float(np.std(pitches))
        pitch_range = float(np.max(pitches) - np.min(pitches))
        
        # Detect oscillations in pitch
        d = np.diff(pitches)
        zero_crossings = np.sum((d[:-1] * d[1:]) < 0)
        
        # No nodding: low variation, no oscillations, or very small range
        if pitch_std < 1.0 and pitch_range < 3.0:
            # Very still, no nodding
            return 80.0
        elif pitch_std < 1.5 and pitch_range < 5.0 and zero_crossings < 2:
            # Minimal movement, no clear nodding pattern
            return 60.0 + (1.5 - pitch_std) / 0.5 * 15.0  # 60-75
        elif zero_crossings < 1:
            # No oscillations detected
            return 50.0
        
        # Nodding detected = low resistance score
        return 15.0

    def _g3_narrowed_pupils(self, cur: dict) -> float:
        ear = cur.get("ear", 0.2)
        if ear < 0.14:
            return 50.0 + (0.14 - ear) / 0.14 * 40.0
        return 20.0

    # ----- Group 4 -----
    def _g4_relaxed_exhale(self, buf: list) -> float:
        # Relaxed exhale: decrease in facial tension + mouth opening
        if len(buf) < 10:
            return 50.0
        
        # Get face scales for normalization
        scales_now = [b.get("face_scale", 50.0) for b in buf[-3:]]
        scales_before = [b.get("face_scale", 50.0) for b in buf[-10:-5]]
        avg_scale_now = float(np.mean(scales_now)) if scales_now else 50.0
        avg_scale_before = float(np.mean(scales_before)) if scales_before else 50.0
        
        # Normalize variance by face scale squared
        var_now_raw = np.mean([b["face_var"] for b in buf[-3:]])
        var_before_raw = np.mean([b["face_var"] for b in buf[-10:-5]])
        var_now = var_now_raw / max(avg_scale_now * avg_scale_now, 1e-6)
        var_before = var_before_raw / max(avg_scale_before * avg_scale_before, 1e-6)
        
        mar_now = np.mean([b["mar"] for b in buf[-3:]])
        mar_before = np.mean([b["mar"] for b in buf[-10:-5]])
        
        # Relaxed exhale: tension decrease (variance drop) + mouth opening (MAR increase)
        tension_drop = var_before > 0 and var_now < var_before * 0.7
        mouth_opening = mar_before > 0 and mar_now > mar_before * 1.05
        
        if tension_drop and mouth_opening:
            # Strong relaxed exhale signal
            return 75.0 + min(20.0, (mar_now / mar_before - 1.05) * 100.0)  # 75-95
        elif tension_drop:
            # Just tension decrease
            return 60.0 + min(15.0, (1.0 - var_now / var_before) * 30.0)  # 60-75
        return 45.0

    def _g4_fixed_gaze(self, buf: list, w: int, h: int) -> float:
        if len(buf) < 8:
            return 50.0
        gx = [b["gaze_x"] for b in buf[-10:]]
        gy = [b["gaze_y"] for b in buf[-10:]]
        std = np.sqrt(np.var(gx) + np.var(gy))
        # Normalize by average face scale for size invariance
        scales = [b.get("face_scale", 50.0) for b in buf[-10:]]
        avg_scale = float(np.mean(scales)) if scales else 50.0
        std_normalized = std / max(avg_scale, 1e-6)
        # Thresholds: 15px -> 30%, 35px -> 70% of inter-ocular distance
        if std_normalized < 0.30:
            return 70.0
        if std_normalized < 0.70:
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
