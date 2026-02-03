# Engagement Metrics: Mathematical Logic & Score Calculation

This document describes how each of the 30 engagement signifier metrics is calculated, including the mathematical formulas, geometric primitives, and **cutting-edge psychology research** underlying each metric. All metrics produce a score in the range 0–100, where **0 = not detected / weak** and **100 = strongly detected**.

---

## 0. Research Basis & Key References

Metrics align with contemporary psychology of facial cues for predicting engagement:

| Area | Key Research | Application |
|------|--------------|-------------|
| **Duchenne smile** | FACS AU 6 + AU 12; Girard et al. (2021) | Genuine vs posed smile; corner lift primary, squinch secondary |
| **Eye contact** | Gaze cueing effect; shared signal hypothesis; eye-mind hypothesis (2024) | Direct gaze = engagement; aversion = disengagement or cognitive load |
| **Eyebrow flash** | Ekman; cross-cultural recognition; ~200ms duration | Interest, acknowledgment, greeting |
| **Head tilt (lateral)** | Davidenko et al. (2018, UC Santa Cruz); Psychology Today | Tilt 11°+ facilitates engagement; exposes neck (trust) |
| **Forward lean** | Riskind & Gotay (1982); embodied approach motivation (2024) | Lean toward = desire, approach motivation |
| **Nodding** | Wells & Petty (1980); self-validation in persuasion | Nodding = agreement, engagement |
| **Lip compression** | FACS AU 23/24; pursed lips = disapproval, restraint | Resistance, withholding, emotional suppression |
| **Contempt** | Ekman & Friesen; unilateral lip curl; **low cross-cultural agreement** (context-dependent) | Asymmetric mouth = contempt; **high false positive risk**—require pronounced asymmetry |
| **Gaze aversion** | **Dual function:** (1) cognitive processing/memory retrieval (2024), (2) disengagement (Cognition, 2017) | **Brief** = processing; **sustained** = disengagement. Duration-based scoring |
| **Lip compression** | FACS AU 23/24; **context-dependent** in professional settings (concentration vs. disapproval) | Require extreme compression; can indicate controlled speech |
| **Cognitive load** | Furrowed brow (AU 4), eye squint; look-up-left (NLU) | Thinking, evaluating, processing |
| **Facial symmetry** | Bilateral symmetry and perceived trustworthiness | Balanced expression = focused engagement |
| **Pupil/eye dilation** | Hess (1965); arousal proxy via eye openness | Wider eyes = interest, arousal (proxy) |

---

## 1. Overview

### 1.1 Metric Groups

| Group | Name | Metrics | Meaning |
|-------|------|---------|---------|
| **G1** | Interest & Engagement | 10 | Positive signals: interest, openness, attentiveness |
| **G2** | Cognitive Load | 7 | Thinking, evaluating, processing (higher = more load) |
| **G3** | Resistance & Objections | 9 | Negative signals: contempt, skepticism, disengagement |
| **G4** | Decision-Ready | 3 | Signals of readiness to commit or close |

### 1.2 Output Scaling (Binary 0 or 100)

All raw scores are produced on an internal scale where **50 = neutral**. Before display, they are converted to **binary 0 or 100**:

```
scaled = max(0, min(100, (raw - 50) × 2))
display = 100 if scaled >= 25 else 0
```

So: raw ≤ 62.5 → 0; raw > 62.5 → 100. Metrics update in real time as expressions change.

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

### 1.4 Noise Robustness

Metrics are made robust to landmark jitter while preserving real-time binary (0/100) updates:

- **Feature-level smoothing**: Scalars (EAR, MAR, gaze, head pose, etc.) are blended with the previous frame (α=0.75) before use, reducing landmark jitter.
- **Binary display** (no smoothing; metrics are 0 or 100 only): Each metric’s 0–100 display value updates in real time; no smoothing (0 or 100 only).
- **Baseline adaptation**: Baselines (EAR, MAR, eye area, Z) use slower EMA (0.92/0.08) so they track the person over time without chasing single-frame noise.
- **Robust aggregation**: Where useful, median of recent frames is used instead of mean (e.g. stillness, eye-contact head pose) to reject outlier frames.
- **Value clamping**: EAR and MAR are clamped to plausible ranges to avoid extreme values from noisy landmarks.

### 1.5 False Positive Reduction

To minimize false positives (incorrectly flagging normal expressions as engagement signals), multiple techniques are applied throughout:

| Technique | Description | Applied To |
|-----------|-------------|------------|
| **Baseline Comparison** | Compare current value to person's typical (neutral) state rather than absolute thresholds | Pupil dilation, Eye squint, Narrowed pupils, Lowered brow, Contempt |
| **Temporal Consistency** | Require signal to persist for multiple frames (2-4+); single-frame spikes ignored | Head tilt, Forward lean, Parted lips, Rhythmic nodding, Eyebrow flash, Eye squint, Thinking brow, Look up/lr, Lip pucker, Contempt |
| **Median Aggregation** | Use median of recent frames instead of current value to filter outliers | Head tilt, Forward lean, Pupil dilation, Eye squint, Look up/lr, Lip pucker, Gaze aversion |
| **Stricter Thresholds** | Require larger deviations from neutral to trigger high scores | All G3 (Resistance) metrics, Fixed gaze, Relaxed exhale |
| **Warmup Period** | Return neutral score until sufficient history (8-16 frames) is available | Contempt, No-nod, Stillness, Fixed gaze, Rhythmic nodding |
| **Head Pose Correction** | Subtract expected asymmetry from head tilt to avoid flagging pose-induced artifacts | Contempt |
| **Duration Checks** | Distinguish brief cognitive processing from sustained disengagement | Gaze aversion, Eye block |
| **Corroboration** | Require multiple signals (e.g., tight lips AND corners down) | Jaw clench, Relaxed exhale |

---

## 1.6 Engagement Cues in Business Meetings: Research Mapping

Recent research (2022–2025) on co-located and online meetings links engagement state to facial and speech cues. The table below maps these findings to the application’s metric groups (G1–G4) and speech-tag categories used for multimodal composite detection.

| Research cue / finding | Metric(s) / group | Speech-tag category | Composite / meaning |
|------------------------|-------------------|----------------------|----------------------|
| **Gaze stability** (3+ s direct gaze; social gaze while listening) | G1 (eye contact), G4 (fixed gaze) | — | Decision-ready; sustained attention |
| **Gaze aversion** (brief = processing; sustained = disengagement) | G2 (look up/lr), G3 (gaze aversion) | confusion, concern | Cognitive overload; disengagement |
| **Nodding** (agreement, self-validation; Wells & Petty) | G1 (rhythmic nodding) | commitment, interest | Decision-readiness; alignment |
| **Duchenne smile + forward lean + eye contact** | G1 (Duchenne, forward lean, eye contact) | interest, commitment | Genuine interest; buying signal |
| **Furrowed brow (AU 4)** (effortful thinking vs. frustration) | G2 (lowered brow, stillness) | confusion, concern | Cognitive overload; need clarity |
| **Lip compression / jaw clench** (withholding, disapproval) | G3 (lip compression, jaw clench) | objection, concern | Skepticism; objection moment |
| **Contempt / asymmetric mouth** (unilateral curl) | G3 (contempt) | objection | Resistance; skepticism |
| **Relaxed exhale + mouth opening** (tension release) | G4 (relaxed exhale) | commitment | Decision-ready; ready to close |
| **Eyebrow flash** (~200 ms; recognition, “yes”) | G1 (eyebrow flash) | interest, realization | Aha moment; acknowledgment |
| **Speaking activity** (who speaks; multimodal > single modality) | — | All phrase categories | Multimodal composites (facial + speech) |
| **Hesitation / uncertainty language** | G2 (cognitive load) | confusion, concern | Cognitive overload; need clarity |
| **Commitment / agreement language** | G4 (decision-ready) | commitment, interest | Decision-readiness; closing window |
| **Objection / pushback language** | G3 (resistance) | objection | Skepticism; objection moment |
| **Realization phrases** (“aha”, “got it”, “that makes sense”) | G1 (spike, eyebrow flash) | interest, realization | Aha/insight moment |

**Speech-tag categories** (from `PHRASE_CATEGORIES` in `services/insight_generator.py`): `objection`, `interest`, `confusion`, `commitment`, `concern`, `timeline`, `budget`, `realization` (added for multimodal composites). Multimodal composites require both facial conditions and a matching recent speech tag (within a short time window) to reduce false positives and align with research on combined cues.

---

## 1.7 Composite Features (Multimodal)

Composite features are **combinations** of metric groups (G1–G4) and, for multimodal composites, **recent speech tags** from the meeting partner’s transcript. When a composite fires (and cooldown has passed), the app triggers an insight popup and optional TTS via Azure OpenAI. Below: trigger conditions and what each composite denotes.

### Multimodal composites (facial + speech; checked first)

| Composite ID | Trigger conditions | Denotes |
|--------------|--------------------|--------|
| **decision_readiness_multimodal** | G4 ≥ 60, G1 ≥ 56, G3 ≥ 56 **and** recent speech tag in `commitment` or `interest` | Partner’s words and face both signal readiness; ask for next step or confirmation. |
| **cognitive_overload_multimodal** | G2 ≥ 56, G1 &lt; 54 **and** recent speech tag in `confusion` or `concern` | Face and language both indicate overload; pause, simplify, or ask what would clarify. |
| **skepticism_objection_multimodal** | G3 raw ≥ 48 **and** recent speech tag in `objection` or `concern` | Verbal objection/concern plus resistant face; address directly and listen. |
| **aha_insight_multimodal** | G2 (recent) ≥ 52, G1 ≥ 58, G3 ≥ 54 **and** recent speech tag in `interest` or `realization` | “Got it” / “that makes sense” plus positive expression; reinforce or deepen. |
| **disengagement_multimodal** | G1 &lt; 46, G4 &lt; 50, G3 raw ≥ 44 **and** no recent `commitment` or `interest` tag | Low engagement cues and no recent positive language; re-engage with a question or shift. |

### Facial-only composites (priority after multimodal)

| Composite ID | Trigger conditions | Denotes |
|--------------|--------------------|--------|
| **closing_window** | G4 ≥ 60, G4 rising (Δ ≥ 16), G3 ≥ 55 | Face has settled; internal decision made; close or ask for commitment. |
| **decision_ready** | G4 ≥ 62, G1 ≥ 56, G3 ≥ 58 | Sustained gaze, relaxed brow, low resistance; ready to commit. |
| **ready_to_sign** | G4 ≥ 65, G2 &lt; 50, G3 ≥ 56 | High commitment, low cognitive load; propose next step. |
| **buying_signal** | G1 ≥ 60, G4 ≥ 52, G3 ≥ 58 | Duchenne, engagement, decision cues; go deeper or reinforce value. |
| **commitment_cue** | G4 ≥ 58, sustained G4 (4-frame mean ≥ 56), G3 ≥ 56 | Nodding, eye contact, relaxed face; summarize and confirm. |
| **cognitive_overload_risk** | G2 ≥ 58, G1 &lt; 52 | Furrowed brow, gaze aversion; System 2 maxed; simplify or pause. |
| **confusion_moment** | G2 ≥ 58 and (G3 raw ≥ 48 or G2 spike) | Processing hit a wall; ask what would clarify or recap. |
| **need_clarity** | G2 ≥ 55, G1 &lt; 56 | Moderate load, low engagement; pause and summarize or ask what’s unclear. |
| **skepticism_surface** | G3 raw ≥ 50 and G3 raw rising | Resistance rising; name it and address. |
| **objection_moment** | G3 raw ≥ 52 | Tension, lip compression; invite them to voice it. |
| **resistance_peak** | G3 raw ≥ 56 | High resistance; acknowledge, don’t push. |
| **hesitation_moment** | G2 ≥ 54, G3 raw ≥ 46 | On the fence; lower stakes or ask what’s holding them back. |
| **disengagement_risk** | G1 &lt; 46, G3 raw ≥ 48 | Attention slipping; re-engage with a question or shift. |
| **objection_fading** | G3 (recent mean) &lt; 52, current G3 ≥ mean + 10 | Tension easing; reinforce value and suggest next step. |
| **aha_moment** | G2 (recent) ≥ 55, G1 ≥ 58, G3 ≥ 54 | Insight just landed; capitalize. |
| **re_engagement_opportunity** | G1 was &lt; 45, now G1 ≥ 50 | Attention returned; strike with key point or question. |
| **alignment_cue** | G1 and G4 both risen ≥ 10 over window | Interest and commitment rising together; propose next step. |
| **genuine_interest** | G1 ≥ 58, G3 ≥ 56 | Duchenne, forward lean, eye contact; genuine curiosity. |
| **listening_active** | G1 ≥ 55, G3 ≥ 54 | Eyes locked, face open; deliver key point. |
| **trust_building_moment** | G1 ≥ 54, G3 ≥ 58 (optional: facial symmetry ≥ 55) | Open face, steady gaze; be transparent. |
| **urgency_sensitive** | G4 ≥ 56, G2 ≥ 50, G3 ≥ 54 | Alert to timing; give clear dates or ask timeline. |
| **processing_deep** | G2 ≥ 56, G3 ≥ 58 | Stillness, inward gaze; give space then reinforce one point. |
| **attention_peak** | G1 ≥ 60 (or G1 ≥ 62 without eye-contact boost) | Max attention; deliver main message now. |
| **rapport_moment** | G1 ≥ 54 (or G1 ≥ 56 without symmetry) | Genuine smile, mirroring; leverage rapport. |

**Detection flow:** Video → G1–G4 (and signifiers); transcript → phrase match → speech tag appended to ring buffer. `detect_opportunity()` is called with group means, history, signifiers, and `recent_speech_tags` (last 12 s). Multimodal composites are evaluated first; facial-only composites follow. First composite that fires and is off cooldown (32 s per type) sets the pending alert; `GET /engagement/state` consumes it and generates the insight (Azure OpenAI + stock fallback). See [DOCUMENTATION.md](DOCUMENTATION.md) §7 and `utils/b2b_opportunity_detector.py`.

---

## 2. Group 1: Interest & Engagement (G1)

### 2.1 Duchenne Marker (g1_duchenne)

**Meaning:** Authentic smile spanning 0–100: low when neither corner lift nor squinch, medium when only one, high when both.

**Research basis:** FACS AU 6 (cheek raise/orbicularis oculi) + AU 12 (lip corner pull/zygomaticus major). Girard et al. (2021): mouth characteristics primary, squinch secondary. Genuine smiles: coordinated eye+mouth; posed: mouth-only.

**Inputs:** mouth landmarks, EAR, face_scale, baseline_ear

**Logic:**
- **AU 12 (corner lift):** `corner_lift = upper_y − (ly + ry)/2`, `lift_ratio = corner_lift / face_scale`. Threshold 0.008; intensity 0–1 mapped from 0.008–0.058.
- **AU 6 (squinch):** When `0.78 ≤ EAR/baseline_ear ≤ 0.96`, intensity 0–1 (peak at 0.87).
- **Combine:** `corner_contrib = au12 × 25`, `squinch_contrib = au6 × 25`, `synergy = min(25, au12 × au6 × 30)` when both > 0.2.
- **Raw score:** `50 + corner_contrib + squinch_contrib + synergy` (capped 50–100).
- **Display:** `(raw − 50) × 2` → 0–100.

**Output:** Raw 50–100; display 0–100 (low/medium/high as above).

---

### 2.2 Pupil Dilation (g1_pupil_dilation)

**Meaning:** Proxy for arousal/interest via eye openness vs. baseline (no direct pupil measurement).

**Research basis:** Hess (1965): pupil dilation correlates with interest/arousal. Since we lack pupil measurement, we use eye-area ratio vs baseline as proxy; wider eyes = higher arousal/interest.

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

**Research basis:** Ekman; cross-cultural eyebrow flash (~200ms) signals recognition, openness, wish to approach. During conversation: nonverbal "yes," interest. Deliberate vs spontaneous differ in speed/duration.

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

**Research basis:** Eye-mind hypothesis; shared signal hypothesis (gaze + expression). Direct gaze = engagement; social inclusion delays disengagement from direct gaze. Leaders/followers differ in gaze patterns.

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

**Meaning:** Lateral tilt (roll); engagement/interest signal per psychology research (Davidenko et al., 2018; UC Santa Cruz). Tilt exposes neck (trust), signals active listening vs passive; tilt as small as 11° facilitates social engagement.

**Inputs:** head_roll (degrees, absolute value)

**Logic (piecewise, research-aligned):**
- `roll < 2.5°`: no tilt (neutral/passive) → raw 50 (display 0)
- `2.5° ≤ roll < 6°`: subtle tilt onset → ramp 50→68
- `6° ≤ roll ≤ 18°`: optimal engagement band (peak ~11°) → 72–98
- `18° < roll ≤ 28°`: strong tilt → ramp down 68→50
- `28° < roll ≤ 40°`: very strong (possible confusion) → 45–50
- `roll > 40°`: extreme ("too weird" per research) → 45

**Output:** Raw 45–98; display 0 when no tilt.

---

### 2.6 Forward Lean (g1_forward_lean)

**Meaning:** Leaning toward camera (engagement).

**Research basis:** Riskind & Gotay; embodied approach motivation. Leaning forward increases left frontal cortical activation to appetitive stimuli; amplifies approach motivation and goal pursuit. Forward lean = desire, interest.

**Inputs:** face_z (mean landmark Z), baseline_z

**Logic:**
- If `z < baseline × 0.99`: `50 + min(52, (1 − z/baseline) × 220)`
- Else: 50

**Output:** 50–102 (raw), effectively capped at 100.

---

### 2.7 Facial Symmetry (g1_facial_symmetry)

**Meaning:** Bilateral symmetry of eyes, eyebrows, mouth, nose.

**Research basis:** Bilateral symmetry associated with perceived trustworthiness, health. Balanced expression = focused engagement; asymmetry can indicate discomfort, distraction, or negative affect. Natural asymmetry 3–8%.

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

**Research basis:** Wells & Petty (1980): nodding increases persuasion (self-validation). Nodding while hearing strong arguments boosts agreement. Part of nonverbal behaviors (with smile, eye contact) that influence attitude change.

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

**Research basis:** Open mouth = receptivity, listening, preparation to speak. Closed = withholding; parted = engaged. Moderate MAR (0.10–0.36) = attentive listening.

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

**Research basis:** FACS AU 4 absent = relaxed. Low brow variance + even inner/outer = not furrowed. Furrowed brow = concentration or negative affect; relaxed = open engagement.

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

**Research basis:** Gaze aversion during cognitive load; look-up-left (NLU) associated with visual/constructed imagery retrieval. Eye-mind link: where people look reflects cognitive processing. Upward/lateral gaze = thinking, accessing memory.

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

**Research basis:** FACS AU 23/24; pursed lips = disapproval, concentration, emotional restraint. Can signal disgust, distrust, or deep thought. Combined with narrowed eyes = skepticism.

**Inputs:** MAR, mouth_width, face_scale

**Logic:**
- `mw_normalized = mouth_width / face_scale`
- Requires both high MAR and narrow mouth.
- Bands: normal mouth (low score), slight pucker (28–50), moderate (50–70), strong (86–100) based on MAR and mw_normalized thresholds.

**Output:** ~5–100 (raw).

---

### 3.3 Eye Squinting (g2_eye_squint)

**Meaning:** Narrowed eyes (concentration, skepticism).

**Research basis:** Narrowed eyes strongly associated with skepticism, distrust, evaluation. Squinting = blocking out or scrutinizing; FACS AU 7 (lid tightener). Cognitive load increases squinting.

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

**Research basis:** One brow raised = curiosity, skepticism, or concentration. Different from bilateral flash; sustained asymmetry = evaluation, doubt. Asymmetric brow = "thinking."

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

**Research basis:** Low face_var = minimal landmark movement. Can indicate focused attention or frozen/withdrawn state. Context-dependent: stillness can signal absorption or disengagement.

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

**Research basis:** FACS AU 4 (brow lowerer/corrugator). Action-unit imposter: downward head tilt can mimic brow furrow. Furrowed brow = concentration, cognitive load, or frustration. AU 4 central to perceived dominance/anger.

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

**Critical note (false positive reduction):** G3 metrics signal resistance, skepticism, or disengagement. **Cutting-edge research** (2023–2024) shows that many of these cues are **context-dependent** and prone to **false positives** in automated detection:

- **Contempt**: Low cross-cultural agreement; context-dependent (not universal). Automated systems show biases.
- **Lip compression (AU23/24)**: Can indicate concentration or controlled speech in professional settings, not only disapproval.
- **Gaze aversion**: Serves **dual function**—(1) internal cognitive processing/memory retrieval (positive), (2) disengagement (negative). Brief gaze shifts (<5 frames) are often **cognitive processing**, not resistance. Only sustained aversion (>6–8 frames) reliably signals disengagement.
- **Furrowed brow (AU4)**: Can signal cognitive load (G2), problem understanding (communicative), or negative affect (G3). Interpret in combination with other signals.

To reduce **unnecessary advice** from erroneous detections, G3 metrics use **stricter thresholds**, **duration checks** (sustained vs. brief), and **corroboration** (e.g. lip compression + corners down for jaw clench). Default scores are lowered when cues are ambiguous.

---

### 4.1 Contempt (g3_contempt)

**Meaning:** Asymmetric mouth (unilateral lip curl) or contempt emotion (Azure). **Rare in professional meetings;** high false positive risk.

**Research basis:** Ekman & Friesen: contempt = unilateral lip curl. **Critical:** Low cross-cultural agreement; automated detection has high false positive risk. Uses **(1) baseline adaptation**, **(2) head-roll correction**, **(3) temporal consistency**, **(4) strict absolute thresholds**.

**Inputs:** mouth corner Y, face_scale, roll, buffer (mouth_corner_asymmetry_ratio), or Azure contempt

**Logic (Landmark path):**
- `asymmetry_ratio = |ly − ry| / face_scale`
- **Roll correction:** if |roll| > 4°, subtract `min(0.10, roll × 0.004)` (tilt creates apparent asymmetry)
- **Baseline:** median of last 18 asymmetry ratios (person's typical neutral asymmetry)
- **Conditions for high score:** (a) corrected_ratio > 0.14, (b) corrected_ratio > baseline + 0.06, (c) ≥2 of last 3 frames also met (a)+(b)
- **Score:** 50 + min(32, excess × 250), capped 50–82; else 35 (neutral/low)
- **Warmup:** return 35 until ≥10 frames in history

**Logic (Azure path):**
- contempt ≤ 0.45 → 35; 0.45–0.55 → 42–50; > 0.55 → 52–82 (scale gently to avoid persistent 100)

**Output:** 35–82 (raw). Only pronounced, sustained contempt above baseline scores high.

---

### 4.2 Nose Crinkle (g3_nose_crinkle)

**Meaning:** Nose shortening (disgust, skepticism). Can also indicate concentration; require **sustained** and **large** shortening.

**Research basis:** FACS AU 9 (nose wrinkler); levator labii activates. Nose crinkle = disgust, skepticism. **Caution:** Can also indicate concentration or strong stimulus. Require **larger** drop (8%+) to reduce false positives.

**Inputs:** nose_height, recent nose heights (last 6 frames)

**Logic:**
- `avg = mean(recent nose_height)`, compare current to avg.
- If `nh < avg×0.88` (stricter: 12% drop, was 8%): `52 + min(36, (1−nh/avg)×90)`
- If `nh < avg×0.94` (6%+ drop): `28 + (0.94−nh/avg)×200`
- Else: 6 (default low)

**Output:** 6–88 (raw).

---

### 4.3 Lip Compression (g3_lip_compression)

**Meaning:** Tightly pressed lips (resistance, withholding). **Caution:** Can also indicate concentration or controlled speech in professional contexts.

**Research basis:** FACS AU 23/24. Pursed/compressed lips = disapproval, restraint. **Critical:** In professional settings, lip compression can also indicate concentration or controlled speech. Require **extreme** compression only; use **more conservative thresholds**.

**Inputs:** MAR_inner (or MAR)

**Logic:**
- `MAR_inner < 0.035` (extreme, was 0.045): 88
- `0.035 ≤ MAR < 0.055`: `65 + (0.055−MAR)/0.020 × 20`
- `0.055 ≤ MAR < 0.08`: `38 + (0.08−MAR)/0.025 × 22`
- `0.08 ≤ MAR < 0.12`: `18 + (0.12−MAR)/0.04 × 16`
- Else: 8 (default low)

**Output:** 8–88 (raw).

---

### 4.4 Eye Block (g3_eye_block)

**Meaning:** Prolonged eye closure (aversion, shutting out). **Caution:** Normal blinks last 1–3 frames; only score high when **prolonged** (>12 frames ≈ 400 ms).

**Research basis:** Extended eye closure = blocking visual input, disengagement, or distress. **Critical:** Normal blinks last 1–3 frames. Only **prolonged** closure (>12 frames at 30 fps ≈ 400 ms) reliably signals blocking; shorter runs are likely blinks.

**Inputs:** EAR over buffer (backward)

**Logic:**
- Count consecutive frames with `EAR < 0.10` from current backward.
- `run ≥ 18` (~600 ms): 88
- `12 ≤ run < 18` (~400–600 ms): `55 + (run−12)/6 × 28`
- `8 ≤ run < 12`: `28 + (run−8)×4`
- Else: 6 (brief = likely blink)

**Output:** 6–88 (raw).

---

### 4.5 Jaw Clenching (g3_jaw_clench)

**Meaning:** Tight jaw (tension, resistance). **Require BOTH** tight lips **and** corners down to reduce false positives.

**Research basis:** Masseter tension = stress, resistance. Low MAR + corners down = clenched. **Critical:** Concentration can cause slight lip compression alone; require **both** extreme compression **and** corners down (downturned mouth) for high score.

**Inputs:** MAR_inner, mouth corners vs. mid, face_scale

**Logic:**
- `corners_down`: (ly+ry)/2 > mid + face_scale×0.04 (stricter: 4%, was 6%)
- If `MAR < 0.055` **and** corners_down: `78 + (0.055−MAR)×200`, capped at 94
- If `MAR < 0.075` and corners_down: `52 + (0.075−MAR)/0.02 × 22`
- If `MAR < 0.055` alone: `48 + (0.055−MAR)/0.02 × 18`
- If `MAR < 0.09`: `28 + (0.09−MAR)/0.035 × 18`
- Else: 8

**Output:** 8–94 (raw).

---

### 4.6 Rapid Blinking (g3_rapid_blink)

**Meaning:** Elevated blink rate (stress, discomfort). **Caution:** Normal rate 10–20/min; only score high when **elevated** (6+ in window).

**Research basis:** Blink rate increases with stress, cognitive load, anxiety. **Critical:** Normal blink rate varies (10–20/min). Rapid blinking can also indicate concentration or dry eyes. Use in combination with other G3 signals; require **higher** count (6+) to reduce false positives.

**Inputs:** blinks in last 2 seconds

**Logic:**
- `b ≥ 6` (raised from 5): 82
- `b ≥ 4` (raised from 3): `48 + (b−4)×15`
- Else: 6 (normal rate)

**Output:** 6–82 (raw).

---

### 4.7 Gaze Aversion (g3_gaze_aversion)

**Meaning:** Looking away. **Critical context dependency:** Gaze aversion serves **dual function**—(1) internal cognitive processing/memory retrieval (positive), (2) disengagement (negative). Only score high when **sustained** (>6–8 frames).

**Research basis:** Gaze aversion = marker of attentional switch to internal world during memory retrieval (2024); occurs within 1 s, lasts ~6 s, linked to retrieval effort. **Brief** gaze shifts (<5 frames) are often **cognitive processing**, not resistance. **Sustained** aversion (>6–8 frames) + extreme deviation reliably signals disengagement. Professional context: off-camera gaze is judged negatively despite cognitive function.

**Inputs:** pitch, yaw, **buffer** (for duration check)

**Logic:**
- `total_dev = √(pitch² + yaw²)` (degrees)
- Count **consecutive** frames (from current backward) with `dev > 8°` (sustained_aversion)
- If sustained_aversion ≥ 8 **and** total_dev > 18°: `55 + min(40, (total_dev−18)/24 × 40)` (long sustained + extreme)
- If sustained_aversion ≥ 6 **and** total_dev > 14°: `38 + min(22, sustained_aversion×2.5)`
- If sustained_aversion ≥ 4 **and** total_dev > 20°: `28 + min(18, (total_dev−20)/15 × 18)` (brief but extreme)
- If total_dev > 25° (current extreme, not sustained): `18 + min(12, (total_dev−25)/20 × 12)`
- Else: 6 (brief or moderate = likely cognitive processing)

**Output:** 6–95 (raw). **Duration-based:** brief aversion scores low; sustained scores high.

---

### 4.8 No-Nod (g3_no_nod)

**Meaning:** Absence of vertical head movement (disengagement, resistance).

**Research basis:** Nodding = agreement (Wells & Petty); absence = disengagement, resistance, passive listening. Low pitch variance + few zero-crossings = no nodding rhythm.

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

**Meaning:** Proxy via eye squint (EAR)—skepticism, negative arousal.

**Research basis:** Pupil constriction correlates with negative arousal; we proxy via narrowed eyes (lower EAR). Squint = skepticism, negative affect. Hess: pupil size reflects arousal valence.

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

**Research basis:** Tension drop (face_var decrease) + mouth opening (MAR increase) = physiological release, acceptance. Signals relaxation, readiness to close, "letting go."

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

**Meaning:** Stable head orientation = looking at a fixed region (camera or elsewhere). Only scores high when head is truly fixated.

**Research basis:** Low gaze variance = focused attention, decision mode. Uses head orientation (yaw, pitch) variance—where the head points, not face position in frame.

**Inputs:** yaw, pitch over last 12 frames

**Logic:**
- `head_std = √(std(yaw)² + std(pitch)²)` in degrees
- `head_std < 1.5°`: 85 (very fixed)
- `head_std < 3°`: 72–82
- `head_std < 5°`: 58–70
- `head_std < 8°`: 45–55
- `head_std ≥ 8°`: 42 (looking around)

**Output:** 42–85 (raw). Drops to low when user looks around.

---

### 5.3 Smile to Genuine (g4_smile_transition)

**Meaning:** Sustained genuine smile (Duchenne + stable mouth).

**Research basis:** Sustained Duchenne + stable mouth = authentic positive affect, decision-ready. Transition from neutral to smile = openness, acceptance. Smile onset dynamics matter.

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

## 9. References (Psychology Research)

| Topic | Key References |
|-------|----------------|
| FACS | Ekman & Friesen (2002); Ekman, P. (ed.) *What the Face Reveals* |
| Duchenne smile | Girard et al. (2021); Ekman & Davidson |
| Head tilt | Davidenko et al. (2018), *Perception*; UC Santa Cruz |
| Forward lean | Riskind & Gotay (1982); embodied approach motivation |
| Nodding | Wells & Petty (1980), *JPSP*; self-validation in persuasion |
| Contempt | Ekman & Friesen; unilateral lip curl; **low cross-cultural agreement** (context-dependent); automated detection false positives (PMC10514002, 2023) |
| Lip compression | FACS AU 23/24; pursed lips = disapproval; **context-dependent** in professional settings (concentration vs. disapproval) |
| Gaze aversion | **Dual function:** (1) memory retrieval, attentional switch to internal world (HAL 2024); (2) disengagement (Cognition, 2017). Duration-based interpretation |
| AU4 (furrowed brow) | **Context-dependent:** cognitive load (G2), problem understanding (communicative), or negative affect (G3). Communicative function (2025) |
| False positive reduction | Context dependency (Nature 2024); cognitive load vs. resistance differentiation; professional meeting context |
| Eye contact | Shared signal hypothesis; eye-mind hypothesis |
| Eyebrow flash | Ekman; cross-cultural recognition |
| Pupil dilation | Hess (1965); arousal/interest |
| Cognitive load | Gaze aversion during difficult tasks; brow furrow (AU 4) |

---

*Source: `utils/expression_signifiers.py`*
