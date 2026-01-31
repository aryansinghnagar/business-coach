# Engagement Metrics: Mathematical Logic & Score Calculation

This document describes how each of the 30 engagement signifier metrics is calculated, including the mathematical formulas and geometric primitives used. All metrics produce a score in the range 0–100, where **0 = not detected / weak** and **100 = strongly detected**.

---

## 1. Overview

### 1.1 Metric Groups

| Group | Name | Metrics | Meaning |
|-------|------|---------|---------|
| **G1** | Interest & Engagement | 10 | Positive signals: interest, openness, attentiveness |
| **G2** | Cognitive Load | 7 | Thinking, evaluating, processing (higher = more load) |
| **G3** | Resistance & Objections | 9 | Negative signals: contempt, skepticism, disengagement |
| **G4** | Decision-Ready | 3 | Signals of readiness to commit or close |

### 1.2 Output Scaling

All raw scores are produced on an internal scale where **50 = neutral**. Before display, they are transformed:

```
scaled = max(0, min(100, (raw - 50) × 2))
```

So: raw 50 → 0, raw 75 → 50, raw 100 → 100.

### 1.3 Core Geometric Primitives

These are computed each frame from facial landmarks (MediaPipe 468 or Azure 27):

| Primitive | Formula | Description |
|-----------|---------|-------------|
| **EAR** (Eye Aspect Ratio) | `v / (h + ε)` where v = vertical extent of eye, h = horizontal extent | Eye openness; ~0.2 when closed |
| **MAR** (Mouth Aspect Ratio) | `mouth_height / mouth_width` | Mouth openness |
| **MAR_inner** | `‖p13 − p14‖ / ‖p61 − p17‖` (vertical lip distance / mouth width) | Inner mouth ratio for lip compression |
| **face_scale** | Inter-ocular distance (‖left_eye_center − right_eye_center‖) | Normalization factor for size invariance |
| **head_yaw** | Nose offset from face center, scaled to degrees | Left/right rotation (~−45° to +45°) |
| **head_pitch** | Nose offset from eye center (Y), scaled to degrees | Up/down tilt |
| **head_roll** | `arctan2(ry − ly, rf − lf)` in degrees | Lateral head tilt |
| **gaze_x, gaze_y** | Eye center − frame center | Face position relative to camera |
| **face_var** | Variance of landmark (x, y) positions | Overall facial movement magnitude |
| **eyebrow_l, eyebrow_r** | `eye_center_y − brow_center_y` per side | Eyebrow raise (positive = raised) |

---

## 2. Group 1: Interest & Engagement (G1)

### 2.1 Duchenne Marker (g1_duchenne)

**Meaning:** Authentic smile (mouth corner lift + optional eye squinch).

**Inputs:** mouth landmarks, EAR, face_scale, baseline_ear

**Logic:**
- **Corner lift:** `corner_lift = mid_y − (ly + ry)/2` (center of mouth Y minus average of corner Ys). Positive = corners lifted.
- **Smile intensity:** `smile_intensity = clamp(corner_lift / (face_scale × 0.065), 0, 1)`
- **Squinch (secondary):** When `0.88 ≤ EAR/baseline_ear ≤ 0.98`, slight eye narrowing adds up to 30% to the score.
- **Combined:** `0.70 × smile_intensity + 0.30 × squinch_intensity` when corner_lift > threshold; otherwise a smaller linear term.

**Output:** 30–98 (raw).

---

### 2.2 Pupil Dilation (g1_pupil_dilation)

**Meaning:** Proxy for arousal/interest via eye openness vs. baseline (no direct pupil measurement).

**Inputs:** eye_area, baseline_eye_area, is_blink

**Logic:**
- During blink: return 0.
- Ratio: `r = eye_area / baseline_eye_area`
- Piecewise linear bands:
  - `r ≥ 1.08`: score = 62 + min(38, (r − 1.08) × 320)
  - `1.03 ≤ r < 1.08`: 54 + (r − 1.03) / 0.05 × 8
  - `1.0 ≤ r < 1.03`: 50 + (r − 1.0) / 0.03 × 4
  - `0.97 ≤ r < 1.0`: 46 + (r − 0.97) / 0.03 × 4
  - `0.92 ≤ r < 0.97`: 38 + (r − 0.92) / 0.05 × 8
  - `r < 0.92`: max(0, 38 + (r − 0.92) × 380)

**Output:** 0–100 (raw).

---

### 2.3 Eyebrow Flash (g1_eyebrow_flash)

**Meaning:** Brief eyebrow raise (interest, acknowledgment).

**Inputs:** eyebrow heights over last 8 frames, face_scale

**Logic:**
- Baseline: mean of first half of heights.
- Threshold: `face_scale × 0.022`
- **Raised & returning:** `max_recent > baseline + threshold` and `cur < max_recent − 0.4×threshold` → flash detected.
  - `flash_magnitude = (max_recent − baseline) / (face_scale × 0.10)`
  - Score = 60 + min(38, flash_magnitude × 120)
- **Raised only:** 50 + min(28, raise_magnitude × 70)
- **Small raise:** if `cur > baseline + 0.5×threshold` → 50 + small bonus
- Otherwise: 50

**Output:** 50–98 (raw).

---

### 2.4 Sustained Eye Contact (g1_eye_contact)

**Meaning:** Head orientation toward camera plus face centering; rewards sustained periods of facing the camera.

**Inputs:** yaw, pitch, gaze_x, gaze_y, face_scale, buffer of last 18 frames

**Logic:**
- **Head score:** `100 − min(100, (|yaw|/25)×50 + (|pitch|/20)×50)` — 0° = 100, ~25° yaw or ~20° pitch = 50.
- **Gaze bonus:** `gaze_dist = √(gaze_x² + gaze_y²)`, `gaze_normalized = min(1, gaze_dist / (face_scale×0.45))`, bonus = (1 − gaze_normalized) × 15.
- **Base score:** head_score + gaze_bonus, capped at 100.
- **Sustained bonus:** Count consecutive frames (from current backward) with score > 50:
  - ≥14 frames: +28
  - ≥10: +18 + (n−10)×2.5
  - ≥6: +8 + (n−6)×2.5
  - ≥3: +2 + (n−3)×2
- **Consistency bonus:** if avg_recent > 55, up to +8.

**Output:** 0–100 (raw).

---

### 2.5 Head Tilt (Lateral) (g1_head_tilt)

**Meaning:** Lateral tilt (roll); moderate tilt can signal interest/curiosity.

**Inputs:** head_roll (degrees)

**Logic (piecewise):**
- `3° ≤ roll ≤ 12°`: optimal band; `72 + (1 − |roll−7.5|/5)×26` → 72–98
- `0° ≤ roll < 3°`: `58 + roll×5` → 58–73
- `12° < roll ≤ 22°`: `62 − (roll−12)×2.2` → 62–40
- `roll > 22°`: `max(10, 38 − (roll−22)×1.6)`

**Output:** 10–98 (raw).

---

### 2.6 Forward Lean (g1_forward_lean)

**Meaning:** Leaning toward camera (engagement).

**Inputs:** face_z (mean landmark Z), baseline_z

**Logic:**
- If `z < baseline × 0.99`: `50 + min(52, (1 − z/baseline) × 220)`
- Else: 50

**Output:** 50–102 (raw), effectively capped at 100.

---

### 2.7 Facial Symmetry (g1_facial_symmetry)

**Meaning:** Bilateral symmetry of eyes, eyebrows, mouth, nose.

**Inputs:** landmark positions, face_scale

**Logic:**
- Face center: midpoint of eye centers.
- Mirror right-side features across center; compute horizontal errors:
  - `eye_error`, `brow_error`, `mouth_error`, `nose_alignment_error`
- Vertical: eye, brow, mouth height differences.
- Normalize all errors by face_scale.
- **Horizontal error (75%):** `0.4×eye + 0.3×mouth + 0.2×brow + 0.1×nose`
- **Vertical error (25%):** `0.5×eye + 0.3×mouth + 0.2×brow`
- `total_error = 0.75×horizontal + 0.25×vertical`
- Piecewise mapping: error → score (0% error = 100, 3–6% ≈ 70–90, 6–10% ≈ 50–70, etc.).

**Output:** 0–100 (raw).

---

### 2.8 Rhythmic Nodding (g1_rhythmic_nodding)

**Meaning:** Vertical pitch oscillations (agreement).

**Inputs:** pitch over last 12 frames

**Logic:**
- `pitch_std = std(pitches)`, count zero-crossings in pitch derivative.
- If `pitch_std < 0.6`: return 34 (no movement).
- If `1 ≤ zero_crossings ≤ 4`: `68 + min(12, pitch_std×4)`
- If `pitch_std > 1.2`: 48
- Else: 42

**Output:** 34–80 (raw).

---

### 2.9 Parted Lips (g1_parted_lips)

**Meaning:** Mouth slightly open (engagement, listening).

**Inputs:** MAR (mouth aspect ratio)

**Logic (piecewise):**
- `0.10 ≤ MAR ≤ 0.36`: `48 + (MAR−0.10)/0.26 × 50` → 48–98
- `0.06 ≤ MAR < 0.10`: `38 + (MAR−0.06)/0.04 × 10`
- `0.36 < MAR ≤ 0.52`: `88 − (MAR−0.36)/0.16 × 22`
- Else: `38 + min(12, MAR×25)`

**Output:** 38–98 (raw).

---

### 2.10 Softened Forehead (g1_softened_forehead)

**Meaning:** Relaxed brow (low tension, not furrowed).

**Inputs:** eyebrow landmarks, face_scale

**Logic:**
- **Variance:** `v = var(brow_points)`, `v_normalized = v / face_scale²`
  - Low variance → high var_score (flat brow).
- **Evenness:** inner vs. outer brow height; relaxed = similar → evenness ≈ 1.
- **Raw:** `50 + 0.5×(var_score−50) + 18×evenness`
- Clamp to [30, 95].

**Output:** 30–95 (raw).

---

## 3. Group 2: Cognitive Load (G2)

### 3.1 Look Up Left/Right (g2_look_up_lr)

**Meaning:** Looking up or left/right (thinking, searching).

**Inputs:** pitch, yaw

**Logic:**
- `looking_up = pitch > 3°`, `looking_lr = |yaw| > 6°`
- Both: `50 + 0.5×(up_intensity + lr_intensity)×44`
- Up only: `50 + min(1, pitch/14)×32`
- LR only: `50 + min(1, |yaw|/35)×18`
- Neither: `46 + min(4, |pitch|×0.4 + |yaw|×0.2)`

**Output:** 46–94 (raw).

---

### 3.2 Lip Pucker (g2_lip_pucker)

**Meaning:** Pursed lips (evaluating, thinking).

**Inputs:** MAR, mouth_width, face_scale

**Logic:**
- `mw_normalized = mouth_width / face_scale`
- Requires both high MAR and narrow mouth.
- Bands: normal mouth (low score), slight pucker (28–50), moderate (50–70), strong (86–100) based on MAR and mw_normalized thresholds.

**Output:** ~5–100 (raw).

---

### 3.3 Eye Squinting (g2_eye_squint)

**Meaning:** Narrowed eyes (concentration, skepticism).

**Inputs:** EAR

**Logic:**
- `0.11 ≤ EAR ≤ 0.19`: `50 + (0.19−EAR)/0.08 × 48` (steeper = more squint)
- `EAR < 0.11`: `88 − EAR×25`
- `0.19 < EAR < 0.22`: `32 + (EAR−0.19)×80`
- Else: 35

**Output:** 32–88 (raw).

---

### 3.4 Thinking Brow (g2_thinking_brow)

**Meaning:** Asymmetric brow raise (concentration, skepticism).

**Inputs:** left/right eyebrow Y means, face_scale

**Logic:**
- `d = |left_brow_y − right_brow_y|`, `d_rel = d / face_scale`
- `0.03 ≤ d_rel ≤ 0.20`: 72
- `0.018 ≤ d_rel ≤ 0.28`: `58 + min(12, d_rel×80)`
- Else: 44

**Output:** 44–72 (raw).

---

### 3.5 Chin Stroke (g2_chin_stroke)

**Meaning:** Hand-on-chin gesture (evaluating). **Not implemented** (no hand detection).

**Output:** 0 (constant).

---

### 3.6 Stillness (g2_stillness)

**Meaning:** Low facial movement (focused attention or frozen).

**Inputs:** face_var over last 8 frames, face_scale

**Logic:**
- `m = mean(face_var)`, `m_normalized = m / face_scale²`
- `m_normalized < 0.0014`: 78
- `m_normalized < 0.005`: `62 + (0.005−m_normalized)/0.0036 × 16`
- Else: 44

**Output:** 44–78 (raw).

---

### 3.7 Lowered Brow (g2_lowered_brow)

**Meaning:** Furrowed brow (concentration, cognitive load).

**Inputs:** brow_y, eye_y, face_scale

**Logic:**
- `dist = eye_y − brow_y` (smaller = more furrowed)
- `dist_rel = dist / face_scale`
- `dist_rel < 0.14`: 82
- `0.14 ≤ dist_rel < 0.19`: `66 + (0.19−dist_rel)/0.05 × 14`
- `0.19 ≤ dist_rel < 0.24`: `54 + (0.24−dist_rel)/0.05 × 12`
- Else: 44

**Output:** 44–82 (raw).

---

## 4. Group 3: Resistance & Objections (G3)

### 4.1 Contempt (g3_contempt)

**Meaning:** Asymmetric mouth or contempt emotion (Azure).

**Inputs:** mouth corner Y, face_scale, or Azure contempt

**Logic:**
- If Azure emotions: `min(100, contempt × 150)`
- Else: `asymmetry = |left_corner_y − right_corner_y|`
  - If `asymmetry > face_scale×0.05`: `28 + (asymmetry/face_scale)×720`
  - Else: 18

**Output:** 18–100 (raw).

---

### 4.2 Nose Crinkle (g3_nose_crinkle)

**Meaning:** Nose shortening (disgust, skepticism).

**Inputs:** nose_height, recent nose heights

**Logic:**
- `avg = mean(recent nose_height)`, compare current to avg.
- If `nh < avg×0.92`: `55 + min(38, (1−nh/avg)×95)`
- If `nh < avg×0.97`: `32 + (0.97−nh/avg)×80`
- Else: 12

**Output:** 12–93 (raw).

---

### 4.3 Lip Compression (g3_lip_compression)

**Meaning:** Tightly pressed lips (resistance, withholding).

**Inputs:** MAR_inner (or MAR)

**Logic:**
- `MAR_inner < 0.045`: 90
- `0.045 ≤ MAR < 0.07`: `70 + (0.07−MAR)/0.025 × 18`
- `0.07 ≤ MAR < 0.10`: `42 + (0.10−MAR)/0.03 × 22`
- `0.10 ≤ MAR < 0.16`: `22 + (0.16−MAR)/0.06 × 18`
- Else: 14

**Output:** 14–90 (raw).

---

### 4.4 Eye Block (g3_eye_block)

**Meaning:** Prolonged eye closure (aversion, shutting out).

**Inputs:** EAR over buffer (backward)

**Logic:**
- Count consecutive frames with `EAR < 0.10` from current backward.
- `run ≥ 18`: 90
- `6 ≤ run < 18`: `48 + (run−6)/12 × 40`
- Else: 8

**Output:** 8–90 (raw).

---

### 4.5 Jaw Clenching (g3_jaw_clench)

**Meaning:** Tight jaw (tension, resistance).

**Inputs:** MAR_inner, mouth corners vs. mid, face_scale

**Logic:**
- `corners_down`: (ly+ry)/2 > mid + face_scale×0.06
- `MAR < 0.065`: `80 + (12 if corners_down else 0)`
- `0.065 ≤ MAR < 0.095`: 54 + linear term + (8 if corners_down)
- `0.095 ≤ MAR < 0.12`: 30 + linear term + (6 if corners_down)
- Else: 12

**Output:** 12–96 (raw).

---

### 4.6 Rapid Blinking (g3_rapid_blink)

**Meaning:** High blink rate (stress, discomfort).

**Inputs:** blinks in last 2 seconds

**Logic:**
- `b ≥ 5`: 85
- `b ≥ 3`: `50 + (b−3)×15`
- Else: 10

**Output:** 10–85 (raw).

---

### 4.7 Gaze Aversion (g3_gaze_aversion)

**Meaning:** Looking away (disengagement).

**Inputs:** pitch, yaw

**Logic:**
- `total_dev = √(pitch² + yaw²)`
- `total_dev > 14°`: `32 + min(58, (total_dev−14)/28 × 58)`
- `6° < total_dev ≤ 14°`: `20 + (total_dev−6)/8 × 12`
- `3° < total_dev ≤ 6°`: `14 + (total_dev−3)/3 × 6`
- Else: 10

**Output:** 10–90 (raw).

---

### 4.8 No-Nod (g3_no_nod)

**Meaning:** Absence of vertical head movement (disengagement, resistance).

**Inputs:** pitch over last 14 frames

**Logic:**
- `pitch_std`, `pitch_range`, zero-crossings
- If `pitch_std < 0.7` and `pitch_range < 2.5`: 82 (very still)
- If `pitch_std < 1.2`, `pitch_range < 4`, `zero_crossings < 2`: `62 + (1.2−pitch_std)/0.5 × 14`
- If `zero_crossings < 1`: 52
- Else: 14

**Output:** 14–82 (raw).

---

### 4.9 Narrowed Pupils (g3_narrowed_pupils)

**Meaning:** Proxy via eye squint (EAR) — skepticism, negative arousal.

**Inputs:** EAR

**Logic:**
- `EAR < 0.13`: `52 + (0.13−EAR)/0.13 × 42`
- `0.13 ≤ EAR < 0.17`: `22 + (0.17−EAR)/0.04 × 28`
- Else: 18

**Output:** 18–94 (raw).

---

### 4.10 Mouth Cover (g3_mouth_cover)

**Meaning:** Hand covering mouth. **Not implemented** (no hand detection).

**Output:** 0 (constant).

---

## 5. Group 4: Decision-Ready (G4)

### 5.1 Relaxed Exhale (g4_relaxed_exhale)

**Meaning:** Release of tension (relaxation, acceptance).

**Inputs:** face_var, MAR over recent vs. prior frames

**Logic:**
- `tension_drop`: var_now < var_before × 0.82
- `mouth_opening`: mar_now > mar_before × 1.03
- Both: `76 + min(22, (mar_now/mar_before − 1.03)×120)`
- Tension drop only: `62 + min(18, (1−var_now/var_before)×38)`
- Else: 44

**Output:** 44–98 (raw).

---

### 5.2 Fixed Gaze (g4_fixed_gaze)

**Meaning:** Stable gaze direction (focus, decision mode).

**Inputs:** gaze_x, gaze_y over last 5 frames, face_scale

**Logic:**
- `std = √(var(gaze_x) + var(gaze_y))`
- `std_normalized = std / face_scale`
- `std_normalized < 0.22`: 74
- `std_normalized < 0.50`: `58 + (0.50−std_normalized)/0.28 × 16`
- Else: 38

**Output:** 38–74 (raw).

---

### 5.3 Smile to Genuine (g4_smile_transition)

**Meaning:** Sustained genuine smile (Duchenne + stable mouth).

**Inputs:** MAR over last 10 frames, g1_duchenne

**Logic:**
- `sustained`: mean(MAR[-5:]) > 0.15 and std(MAR[-5:]) < 0.045
- If `sustained` and `duchenne ≥ 58`: 82
- If `duchenne ≥ 54`: `62 + min(12, (duchenne−54)×0.5)`
- Else: 45–48

**Output:** 45–82 (raw).

---

## 6. Group Aggregation & Composite Score

### 6.1 Group Means

Each group G1–G4 is the **weighted mean** of its signifiers (weights from `weights/signifier_weights.json`, default 1.0 each):

```
Gk = Σ(wi × si) / Σ(wi)   for signifiers i in group k
```

For **G3 (Resistance)**, the raw mean is **inverted** before use:

```
G3_display = 100 − G3_raw
```

So high raw resistance → low G3_display (undesirable).

### 6.2 Composite Score (when used)

```
composite = w1×G1 + w2×G2 + w3×G3 + w4×G4
```

Default group weights: `[0.35, 0.15, 0.35, 0.15]`.

**Adjustments:**
- If `(G1 + G4)/2 > 62`: `composite += 8` (interest + decision-ready bonus)
- If `G3_raw > 38`: `composite −= 10` (resistance penalty)

Final: `score = clamp(composite, 0, 100)`.

---

## 7. Reference: Landmark Indices (MediaPipe)

| Region | Indices |
|--------|---------|
| Left eye | 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 |
| Right eye | 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 |
| Mouth | 61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318 |
| Left eyebrow | 107, 55, 65, 52, 53, 46 |
| Right eyebrow | 336, 296, 334, 293, 300, 276 |
| Nose | 4, 6, 19, 20, 51, 94, 168, 197, 326, 327, 358, 359, 360, 361 |
| Mouth corners | 61 (L), 17 (R) |
| Nose tip | 4 |
| Chin | 175 |
| Face extent | 234 (L), 454 (R) |

---

## 8. Temporal Buffer

Many metrics use a **ring buffer** of the last N frames (default 22, 12 in lightweight mode). Each frame stores: EAR, eye_area, eyebrow heights, pitch, yaw, roll, MAR, face_var, gaze, face_scale, etc.

- **Baselines** (eye_area, EAR, face_z) are updated with **exponential moving average** (α ≈ 0.12) when not blinking.
- **Blink counting** resets every 2 seconds; blinks of 2–8 frames duration are counted.

---

*Source: `utils/expression_signifiers.py`*
