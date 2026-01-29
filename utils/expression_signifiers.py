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
        self._pupil_dilation_history: deque = deque(maxlen=3)  # Minimal smoothing for real-time
        self._last_pupil_dilation_score: float = 0.0  # Last valid score (for blink frames); default 0
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
        self._last_pupil_dilation_score = 0.0  # default 0; output scale 0–100
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

        # Baselines (EMA) - faster adaptation for real-time sensitivity to small changes
        if is_blink < 0.5:  # Only update baseline when not blinking
            if self._baseline_eye_area <= 0 and eye_area > 0:
                self._baseline_eye_area = eye_area
            elif eye_area > 0:
                self._baseline_eye_area = 0.88 * self._baseline_eye_area + 0.12 * eye_area
        if self._baseline_z == 0 and face_z != 0:
            self._baseline_z = face_z
        else:
            self._baseline_z = 0.88 * self._baseline_z + 0.12 * face_z if face_z != 0 else self._baseline_z
        # Baseline EAR for Duchenne squinch detection (exclude blinks)
        if is_blink < 0.5:  # Only update baseline when not blinking
            if self._baseline_ear <= 0 and ear > 0:
                self._baseline_ear = ear
            elif ear > 0:
                self._baseline_ear = 0.88 * self._baseline_ear + 0.12 * ear

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
                out[k] = 0.0
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
        out["g1_eye_contact"] = self._g1_eye_contact(cur, w, h, buf)
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
        out["g2_chin_stroke"] = 0.0  # no hand detection
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
        out["g3_mouth_cover"] = 0.0  # no hand detection

        # --- Group 4: Decision-Ready ---
        out["g4_relaxed_exhale"] = self._g4_relaxed_exhale(buf)
        out["g4_fixed_gaze"] = self._g4_fixed_gaze(buf, w, h)
        out["g4_smile_transition"] = self._g4_smile_transition(buf, out.get("g1_duchenne", 0))

        # Scale: default/weak = 0, strong = 100 (map old 50 -> 0, 100 -> 100)
        for k in out:
            v = float(out[k])
            scaled = (v - 50.0) * 2.0
            out[k] = max(0.0, min(100.0, scaled))
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
            return float(np.mean([scores[k] for k in grp_keys if k in scores])) if grp_keys else 0.0

        # When all metrics are 0 (e.g. no data), composite = 0
        if all(float(scores.get(k, 0)) == 0.0 for k in keys):
            return 0.0

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
        # Composite is 0-100: 0 = no engagement, 100 = high (g3 already inverted so high g3 = low resistance)
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
        
        # Calculate smile intensity (gradual, more sensitive to small corner lift)
        ly = float(mouth_pts[0, 1])
        ry = float(mouth_pts[5, 1]) if mouth_pts.shape[0] > 5 else ly
        mid_y = float(np.mean(mouth_pts[:, 1]))
        corner_lift = (mid_y - (ly + ry) / 2)  # Positive = corners up
        smile_threshold = face_scale * 0.018  # Lower threshold: detect smaller smiles
        smile_intensity = min(1.0, max(0.0, corner_lift / max(face_scale * 0.07, 1e-6)))  # Steeper: small lift -> higher intensity
        
        # Calculate squinch: EAR reduction relative to baseline (wider band for sensitivity)
        baseline_ear = self._baseline_ear if self._baseline_ear > 0 else 0.25
        if baseline_ear <= 0:
            return 50.0
        ear_ratio = ear / baseline_ear
        # Squinch: EAR 88-98% of baseline (sensitive to slight narrowing)
        squinch_intensity = 0.0
        if 0.88 <= ear_ratio <= 0.98:
            squinch_intensity = 1.0 - abs(ear_ratio - 0.93) / 0.10  # Peak at 0.93
        elif ear_ratio < 0.88:
            squinch_intensity = 0.5  # Too much squint, less genuine
        
        # Combine: Duchenne = smile + squinch (higher gain for small changes)
        if corner_lift > smile_threshold and squinch_intensity > 0.25:
            combined = 0.6 * smile_intensity + 0.4 * squinch_intensity
            return 50.0 + combined * 52.0  # 50-102 -> clamp 100
        elif corner_lift > smile_threshold:
            return 50.0 + smile_intensity * 38.0  # 50-88
        elif squinch_intensity > 0.25:
            return 45.0 + squinch_intensity * 14.0  # 45-59
        return 28.0 + min(1.0, corner_lift / max(face_scale * 0.04, 1e-6)) * 12.0  # Slight lift still scores 28-40

    def _g1_pupil_dilation(self, cur: dict) -> float:
        # Proxy: eye area vs baseline (wider = higher)
        # Exclude blinks and use temporal smoothing for stability
        is_blink = cur.get("is_blink", 0.0)
        area = cur.get("eye_area", 0.0)
        base = max(1e-6, self._baseline_eye_area)
        
        # During blinks, return last valid score (decay toward 0)
        if is_blink > 0.5:
            smoothed = 0.96 * self._last_pupil_dilation_score + 0.04 * 0.0
            self._last_pupil_dilation_score = smoothed
            return float(smoothed)
        
        # Calculate ratio only when not blinking
        r = area / base if base > 0 else 1.0
        
        # Score: steeper bands so small ratio changes produce larger score changes
        # r < 0.92: narrowed -> 0-38
        # r 0.92-0.97: slightly narrow -> 38-46
        # r 0.97-1.03: normal -> 46-54
        # r 1.03-1.08: slightly wide -> 54-62
        # r > 1.08: dilated -> 62-100
        if r >= 1.08:
            score = 62.0 + min(38.0, (r - 1.08) * 320.0)
        elif r >= 1.03:
            score = 54.0 + (r - 1.03) / 0.05 * 8.0
        elif r >= 1.0:
            score = 50.0 + (r - 1.0) / 0.03 * 4.0
        elif r >= 0.97:
            score = 46.0 + (r - 0.97) / 0.03 * 4.0
        elif r >= 0.92:
            score = 38.0 + (r - 0.92) / 0.05 * 8.0
        else:
            score = max(0.0, 38.0 + (r - 0.92) * 380.0)
        
        # Minimal smoothing: 92% current, 8% recent (low latency)
        self._pupil_dilation_history.append(score)
        if len(self._pupil_dilation_history) >= 2:
            recent_avg = float(np.mean(list(self._pupil_dilation_history)[:-1]))
            smoothed_score = 0.92 * score + 0.08 * recent_avg
        else:
            smoothed_score = score
        
        # Update last valid score
        self._last_pupil_dilation_score = smoothed_score
        
        return float(max(0.0, min(100.0, smoothed_score)))

    def _g1_eyebrow_flash(self, buf: list) -> float:
        # Eyebrow flash: rapid raise and return (shorter window, lower threshold for sensitivity)
        if len(buf) < 8:
            return 50.0
        heights = [(b["eyebrow_l"] + b["eyebrow_r"]) / 2 for b in buf]
        face_scale = buf[-1].get("face_scale", 50.0)
        threshold = face_scale * 0.022  # Lower: detect smaller raises
        
        baseline = float(np.mean(heights[:max(1, len(heights) // 2)]))
        cur = heights[-1]
        recent_heights = heights[-4:]
        max_recent = float(np.max(recent_heights))
        
        raised = max_recent > baseline + threshold
        returning = cur < max_recent - threshold * 0.4
        
        if raised and returning:
            flash_magnitude = (max_recent - baseline) / max(face_scale * 0.10, 1e-6)
            return 60.0 + min(38.0, flash_magnitude * 120.0)
        elif raised:
            raise_magnitude = (cur - baseline) / max(face_scale * 0.10, 1e-6)
            return 50.0 + min(28.0, raise_magnitude * 70.0)
        # Small raise above baseline still scores slightly above 50
        if cur > baseline + threshold * 0.5:
            return 50.0 + min(8.0, (cur - baseline) / max(face_scale * 0.08, 1e-6) * 30.0)
        return 50.0

    def _g1_eye_contact(self, cur: dict, w: int, h: int, buf: list) -> float:
        """
        Sustained Eye Contact: Head orientation toward camera (primary) plus face position (secondary).
        Rewards sustained periods of facing the camera.
        """
        gx = cur.get("gaze_x", 0)
        gy = cur.get("gaze_y", 0)
        yaw = cur.get("yaw", 0)
        pitch = cur.get("pitch", 0)
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

    def _g1_head_tilt(self, cur: dict) -> float:
        # Head tilt: more sensitive bands; small tilt (3-12°) scores high
        roll = abs(cur.get("roll", 0))
        
        if 3 <= roll <= 12:
            optimal_dist = abs(roll - 7.5)
            return 72.0 + (1.0 - optimal_dist / 5.0) * 26.0  # 72-98
        if 0 <= roll < 3:
            return 58.0 + roll * 5.0  # 58-73
        if 12 < roll <= 22:
            return 62.0 - (roll - 12) * 2.2  # 62-40
        return max(10.0, 38.0 - (roll - 22) * 1.6)

    def _g1_forward_lean(self, cur: dict) -> float:
        z = cur.get("face_z", 0)
        base = self._baseline_z
        if base == 0 or z == 0:
            return 50.0
        # More sensitive: smaller lean-in (e.g. 0.98) already boosts score
        if z < base * 0.99:
            return 50.0 + min(52.0, (1.0 - z / base) * 220.0)
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
        # Rhythmic nodding: vertical head movement (pitch) showing agreement
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

    def _g1_parted_lips(self, cur: dict) -> float:
        mar = cur.get("mar", 0.2)
        # Steeper: small MAR change -> larger score change
        if 0.10 <= mar <= 0.36:
            return 48.0 + (mar - 0.10) / 0.26 * 50.0  # 48-98
        if 0.06 <= mar < 0.10:
            return 38.0 + (mar - 0.06) / 0.04 * 10.0
        if 0.36 < mar <= 0.52:
            return 88.0 - (mar - 0.36) / 0.16 * 22.0
        return 38.0 + min(12.0, mar * 25.0)

    def _g1_softened_forehead(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        # Low brow tension: relaxed = flat brow (low variance) + inner/outer brow even (not furrowed)
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        if len(lb) < 2 or len(rb) < 2:
            return 50.0
        face_scale = max(20.0, cur.get("face_scale", 50.0))

        # 1) Variance of brow points (low = flat = relaxed)
        v = float(np.var(np.vstack([lb[:, :2], rb[:, :2]])))
        v_normalized = v / max(face_scale * face_scale, 1e-6)

        # 2) Inner vs outer brow height: relaxed = similar height; furrowed = inner pulled down
        # Left: inner = point closest to nose (min x), outer = max x
        left_inner_y = float(lb[np.argmin(lb[:, 0]), 1])
        left_outer_y = float(lb[np.argmax(lb[:, 0]), 1])
        right_inner_y = float(rb[np.argmin(rb[:, 0]), 1])
        right_outer_y = float(rb[np.argmax(rb[:, 0]), 1])
        # Furrowed: inner lower than outer (higher y) -> diff > 0
        inner_outer_diff_left = left_inner_y - left_outer_y
        inner_outer_diff_right = right_inner_y - right_outer_y
        diff_normalized = (abs(inner_outer_diff_left) + abs(inner_outer_diff_right)) / max(face_scale * 0.15, 1e-6)
        evenness = max(0.0, 1.0 - min(1.0, diff_normalized))  # 1 = perfectly even, 0 = very furrowed

        # Score from variance: low v_norm -> high (relaxed)
        if v_normalized < 0.0008:
            var_score = 85.0
        elif v_normalized < 0.003:
            var_score = 70.0 + (0.003 - v_normalized) / 0.0022 * 15.0
        elif v_normalized < 0.008:
            var_score = 50.0 + (0.008 - v_normalized) / 0.005 * 20.0
        else:
            var_score = max(30.0, 50.0 - (v_normalized - 0.008) * 1200.0)

        # Combine: 60% variance (flat brow), 40% evenness (inner/outer similar = not furrowed)
        raw = 50.0 + 0.5 * (var_score - 50.0) + 18.0 * evenness
        return float(max(30.0, min(95.0, raw)))

    # ----- Group 2 -----
    def _g2_look_up_lr(self, cur: dict) -> float:
        p, y = cur.get("pitch", 0), cur.get("yaw", 0)
        # Lower thresholds: small look-up/lr scores higher
        looking_up = p > 3
        looking_lr = abs(y) > 6
        
        if looking_up and looking_lr:
            up_intensity = min(1.0, p / 20.0)
            lr_intensity = min(1.0, abs(y) / 32.0)
            combined = (up_intensity + lr_intensity) / 2.0
            return 50.0 + combined * 44.0
        elif looking_up:
            up_intensity = min(1.0, p / 14.0)
            return 50.0 + up_intensity * 32.0
        elif looking_lr:
            lr_intensity = min(1.0, abs(y) / 35.0)
            return 50.0 + lr_intensity * 18.0
        return 46.0 + min(4.0, abs(p) * 0.4 + abs(y) * 0.2)

    def _g2_lip_pucker(self, lm: np.ndarray, cur: dict) -> float:
        """
        Lip Pucker: Detects pursed lips (thinking, evaluating expression).
        
        Lip pucker is characterized by:
        - High MAR (mouth aspect ratio - vertical opening relative to width)
        - Narrow mouth width (lips pursed together)
        - Both conditions must be met for a pucker
        
        Returns 0-100 where:
        - 0-30: No pucker (normal mouth)
        - 30-60: Slight pucker
        - 60-80: Moderate pucker
        - 80-100: Strong pucker
        """
        mar = cur.get("mar", 0.2)
        face_scale = cur.get("face_scale", 50.0)
        mw = cur.get("mouth_w", 40.0)
        
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

    def _g2_eye_squint(self, cur: dict) -> float:
        ear = cur.get("ear", 0.2)
        # Steeper: small EAR drop -> larger score increase
        if 0.11 <= ear <= 0.19:
            return 50.0 + (0.19 - ear) / 0.08 * 48.0
        if ear < 0.11:
            return 88.0 - ear * 25.0  # 85-88 for very narrow
        return 32.0 + (ear - 0.19) * 80.0 if ear < 0.22 else 35.0

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
        d_rel = d / face_scale
        # More sensitive: smaller asymmetry (thinking brow) scores high
        if 0.03 <= d_rel <= 0.20:
            return 72.0
        if 0.018 <= d_rel <= 0.28:
            return 58.0 + min(12.0, d_rel * 80.0)
        return 44.0

    def _g2_stillness(self, buf: list) -> float:
        if len(buf) < 8:
            return 50.0
        vars_list = [b["face_var"] for b in buf[-8:]]
        scales = [b.get("face_scale", 50.0) for b in buf[-8:]]
        avg_scale = float(np.mean(scales)) if scales else 50.0
        avg_scale_sq = max(400.0, avg_scale * avg_scale)
        m = float(np.mean(vars_list))
        m_normalized = m / avg_scale_sq
        if m_normalized < 0.0014:
            return 78.0
        if m_normalized < 0.005:
            return 62.0 + (0.005 - m_normalized) / 0.0036 * 16.0
        return 44.0

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
        
        # More sensitive bands
        if dist_rel < 0.14:
            return 82.0
        elif dist_rel < 0.19:
            return 66.0 + (0.19 - dist_rel) / 0.05 * 14.0
        elif dist_rel < 0.24:
            return 54.0 + (0.24 - dist_rel) / 0.05 * 12.0
        return 44.0

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
            asymmetry = abs(ly - ry)
            if asymmetry > face_scale * 0.05:  # More sensitive: smaller asymmetry scores
                return min(100.0, 28.0 + (asymmetry / face_scale) * 720.0)
        return 18.0

    def _g3_nose_crinkle(self, cur: dict, buf: list) -> float:
        nh = cur.get("nose_height", 0.0)
        if nh <= 0 or len(buf) < 5:
            return 12.0
        recent = [b.get("nose_height", nh) for b in buf[-5:-1] if b.get("nose_height", 0) > 0]
        if not recent:
            return 12.0
        avg = float(np.mean(recent))
        if avg <= 1e-6:
            return 12.0
        if nh < avg * 0.92:  # More sensitive: smaller shortening scores
            return 55.0 + min(38.0, (1.0 - nh / avg) * 95.0)
        if nh < avg * 0.97:
            return 32.0 + (0.97 - nh / avg) * 80.0
        return 12.0

    def _g3_lip_compression(self, cur: dict) -> float:
        mar = cur.get("mar_inner", cur.get("mar", 0.2))
        # Steeper: small MAR drop -> larger score increase
        if mar < 0.045:
            return 90.0
        if mar < 0.07:
            return 70.0 + (0.07 - mar) / 0.025 * 18.0
        if mar < 0.10:
            return 42.0 + (0.10 - mar) / 0.03 * 22.0
        if mar < 0.16:
            return 22.0 + (0.16 - mar) / 0.06 * 18.0
        return 14.0

    def _g3_eye_block(self, buf: list) -> float:
        if len(buf) < 5:
            return 8.0
        run = 0
        for b in reversed(buf):
            if b.get("ear", 0.2) < 0.10:
                run += 1
            else:
                break
        if run >= 18:
            return 90.0
        if run >= 6:  # Shorter run still scores (lower latency)
            return 48.0 + (run - 6) / 12 * 40.0
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
        if mar < 0.065:
            return min(96.0, 80.0 + (12.0 if corners_down else 0.0))
        if mar < 0.095:
            return 54.0 + (0.095 - mar) / 0.03 * 22.0 + (8.0 if corners_down else 0.0)
        if mar < 0.12:
            return 30.0 + (0.12 - mar) / 0.025 * 20.0 + (6.0 if corners_down else 0.0)
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
        
        # More sensitive: smaller deviation scores higher aversion
        if total_dev > 14.0:
            return 32.0 + min(58.0, (total_dev - 14.0) / 28.0 * 58.0)
        elif total_dev > 6.0:
            return 20.0 + (total_dev - 6.0) / 8.0 * 12.0
        elif total_dev > 3.0:
            return 14.0 + (total_dev - 3.0) / 3.0 * 6.0
        return 10.0

    def _g3_no_nod(self, buf: list) -> float:
        if len(buf) < 12:
            return 10.0
        pitches = [b["pitch"] for b in buf[-14:]]
        pitch_std = float(np.std(pitches))
        pitch_range = float(np.max(pitches) - np.min(pitches))
        d = np.diff(pitches)
        zero_crossings = np.sum((d[:-1] * d[1:]) < 0)
        if pitch_std < 0.7 and pitch_range < 2.5:
            return 82.0
        elif pitch_std < 1.2 and pitch_range < 4.0 and zero_crossings < 2:
            return 62.0 + (1.2 - pitch_std) / 0.5 * 14.0
        elif zero_crossings < 1:
            return 52.0
        return 14.0

    def _g3_narrowed_pupils(self, cur: dict) -> float:
        ear = cur.get("ear", 0.2)
        if ear < 0.13:
            return 52.0 + (0.13 - ear) / 0.13 * 42.0
        if ear < 0.17:
            return 22.0 + (0.17 - ear) / 0.04 * 28.0
        return 18.0

    # ----- Group 4 -----
    def _g4_relaxed_exhale(self, buf: list) -> float:
        if len(buf) < 6:
            return 50.0
        scales_now = [b.get("face_scale", 50.0) for b in buf[-2:]]
        scales_before = [b.get("face_scale", 50.0) for b in buf[-6:-3]]
        avg_scale_now = float(np.mean(scales_now)) if scales_now else 50.0
        avg_scale_before = float(np.mean(scales_before)) if scales_before else 50.0
        var_now_raw = np.mean([b["face_var"] for b in buf[-2:]])
        var_before_raw = np.mean([b["face_var"] for b in buf[-6:-3]])
        var_now = var_now_raw / max(avg_scale_now * avg_scale_now, 1e-6)
        var_before = var_before_raw / max(avg_scale_before * avg_scale_before, 1e-6)
        mar_now = np.mean([b["mar"] for b in buf[-2:]])
        mar_before = np.mean([b["mar"] for b in buf[-6:-3]])
        tension_drop = var_before > 0 and var_now < var_before * 0.82  # More sensitive
        mouth_opening = mar_before > 0 and mar_now > mar_before * 1.03
        if tension_drop and mouth_opening:
            return 76.0 + min(22.0, (mar_now / mar_before - 1.03) * 120.0)
        elif tension_drop:
            return 62.0 + min(18.0, (1.0 - var_now / var_before) * 38.0)
        return 44.0

    def _g4_fixed_gaze(self, buf: list, w: int, h: int) -> float:
        if len(buf) < 5:
            return 50.0
        gx = [b["gaze_x"] for b in buf[-5:]]
        gy = [b["gaze_y"] for b in buf[-5:]]
        std = np.sqrt(np.var(gx) + np.var(gy))
        scales = [b.get("face_scale", 50.0) for b in buf[-5:]]
        avg_scale = float(np.mean(scales)) if scales else 50.0
        std_normalized = std / max(avg_scale, 1e-6)
        if std_normalized < 0.22:
            return 74.0
        if std_normalized < 0.50:
            return 58.0 + (0.50 - std_normalized) / 0.28 * 16.0
        return 38.0

    def _g4_smile_transition(self, buf: list, duchenne: float) -> float:
        if len(buf) < 10 or duchenne < 50:
            return 45.0
        mars = [b["mar"] for b in buf[-10:]]
        sustained = np.mean(mars[-5:]) > 0.15 and np.std(mars[-5:]) < 0.045  # More sensitive
        if sustained and duchenne >= 58:
            return 82.0
        if duchenne >= 54:
            return 62.0 + min(12.0, (duchenne - 54) * 0.5)
        return 48.0
