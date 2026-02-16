# Metrics calculation: logic and math

This document explains the **logic and mathematics** behind every score in the engagement pipeline: the **overall engagement score**, **basic metrics**, **core signifiers** (G1–G4), **group means**, and **composite metrics**. All scores are normalized to **0–100** unless noted.

**For a plain-language guide** (what we detect, why it matters, how we score it) suitable for non-technical readers, see [guide.md](guide.md).

**Code references:** `helpers.py` (EngagementScorer, ExpressionSignifierEngine, compute_composite_metrics), `detector.py` (_metrics_from_signifiers, state assembly).

---

## 1. Overall engagement score

The main **engagement score** (0–100) is produced by **EngagementScorer.calculate_score(metrics)** in `helpers.py`. It is **not** a direct average of signifiers; it is computed from the **six basic metrics** with weights and a weakest-link factor.

### Formula

1. **Weighted sum of basic metrics**
   - `raw = attention×0.24 + eye_contact×0.26 + facial_expressiveness×0.14 + head_movement×0.14 + symmetry×0.11 + mouth_activity×0.11`
   - Weights reflect meeting psychology: eye contact and attention dominate.

2. **Weakest-link factor**
   - `min_m = min(attention, eye_contact, facial_expressiveness, head_movement, symmetry, mouth_activity)`
   - `factor = 0.58 + 0.42 × (min_m / 100)`
   - So a single very low metric pulls the score down (softer curve than a pure minimum so one bad metric doesn’t over-penalize).

3. **Final score**
   - `score = clip(raw × factor, 0, 100)`

Invalid/Non-finite metric values are treated as 0 before the calculation.

---

## 2. Basic metrics (six)

Basic metrics are **EngagementMetrics**: attention, eye_contact, facial_expressiveness, head_movement, symmetry, mouth_activity. They are built in **engagement_detector._metrics_from_signifiers()** from **signifier scores** (and optionally blended with **EngagementScorer** when landmarks are available).

### 2.1 Attention

- **Signifier path:** `attention = mean(g1_duchenne, g1_eye_contact, g1_eyebrow_flash, g1_pupil_dilation)`.
- **Hybrid (when landmarks + frame_shape provided):** `attention = 0.5 × att_signer + 0.5 × att_scorer`, where `att_scorer` comes from **EngagementScorer.compute_attention_eye_contact_from_landmarks** → **_compute_attention**.
- **Scorer (EAR-based):** Uses average EAR (eye aspect ratio). Piecewise linear mapping:
  - EAR &lt; 0.02 → 0; 0.02–0.12 → 0–25; 0.12–0.18 → 25–45; 0.18–0.22 → 45–70; **0.22–0.28 → 70–100** (peak band); 0.28–0.38 → 70–55; 0.38–0.55 → 55–25; else lower.
  - Asymmetry penalty: `attention -= (|left_ear − right_ear| / avg_ear) × 25` (capped so attention ≥ 0).

### 2.2 Eye contact

- **Signifier path:** `eye_contact = g1_eye_contact`.
- **Hybrid:** `eye_contact = 0.5 × eye_signer + 0.5 × eye_scorer` (same landmarks path as above).
- **Scorer:** Based on (1) distance of eye center from frame center: `nd = distance / (half diagonal)`, `eye_contact = (1 − nd^1.3)×100`; (2) head yaw/pitch penalty: `yaw_penalty = min(1, yaw_deg/35)×0.5`, `eye_contact *= (1 − yaw_penalty)`.

### 2.3 Facial expressiveness

- **Formula:** `expr_vals = [s[g1_duchenne], s[g1_parted_lips], s[g4_smile_transition], s[g1_softened_forehead], s[g1_eyebrow_flash], s[g1_micro_smile], s[g1_eye_widening], s[g1_brow_raise_sustained]]`
- `expr_mean = mean(expr_vals)`, `expr_max = max(expr_vals)`
- **facial_expressiveness = 0.5×expr_mean + 0.5×expr_max** (then clipped to 100). So one strong expression can raise the metric.

### 2.4 Head movement

- **Formula:** `head_vals = [g1_head_tilt, g1_rhythmic_nodding, g1_forward_lean, (100 − g2_stillness)]`. The last term is “movement amount” (inverse of stillness).
- `head_mean = mean(head_vals)`, `head_max = max(head_vals)`
- **head_movement = min(100, 0.5×head_mean + 0.5×head_max)**.

### 2.5 Symmetry

- **Formula:** `symmetry = g1_facial_symmetry` (from signifiers). No scorer path in the basic-metrics pipeline.

### 2.6 Mouth activity

- **Formula:** `mouth_activity = g1_parted_lips`.

---

## 3. Core signifiers (G1–G4)

Signifiers are computed by **ExpressionSignifierEngine.get_all_scores()** in `helpers.py`. Each signifier is a function of the current frame’s landmarks/cur and a short **buffer** of recent frames. All outputs are clamped to **0–100**.

### 3.1 Group structure

- **G1 (Interest & engagement):** Duchenne, pupil dilation, eyebrow flash, eye contact, head tilt, forward lean, facial symmetry, rhythmic nodding, parted lips, softened forehead, micro smile, brow raise sustained, mouth open receptive, eye widening, nod intensity.
- **G2 (Cognitive load):** Look up L/R, lip pucker, eye squint, thinking brow, stillness, gaze shift frequency, mouth tight (evaluative).
- **G3 (Resistance):** Contempt, nose crinkle, lip compression, eye block, jaw clench, rapid blink, gaze aversion, no-nod, narrowed pupils, lip corner dip, brow lower sustained, eye squeeze, head shake. Stored as raw 0–100; **composites use (100 − G3)** so high = more resistance.
- **G4 (Decision-ready):** Relaxed exhale, fixed gaze, smile transition, mouth relax, smile sustain.

### 3.2 How signifier scores are computed (pattern)

- Each `_g*_*(...)` method returns a **float 0–100**.
- Inputs: `lm` (landmarks), `cur` (current frame dict: EAR, gaze, yaw, pitch, face_scale, mouth/eye metrics, etc.), `buf` (recent frames), and sometimes `fr` (Azure face result).
- Many use **piecewise linear bands** on a geometric quantity (e.g. EAR ratio, MAR, distance, angle) mapped to score bands so that small changes produce visible movement and the full 0–100 range is used.
- **Temporal logic:** e.g. eyebrow flash requires “raise then return”; eye contact uses sustained windows; stillness uses variance over the buffer.

### 3.3 Representative formulas (logic/math)

**g1_duchenne**  
- Combines **AU12** (lip corner lift) and **AU6** (eye squinch).  
- AU12: `corner_lift = upper_y − (ly+ry)/2`, `lift_ratio = corner_lift/face_scale`; intensity from threshold 0.002 with ramp to 1 by ~0.02.  
- AU6: EAR vs baseline; `ear_ratio = ear/baseline_ear`; intensity when ratio in [au6_lo, au6_hi] (default 0.68–1.0).  
- **Score = 18 + AU12_contrib(≈38) + AU6_contrib(≈38) + synergy(up to 24)** when both present.

**g1_eye_contact**  
- **Head:** `head_score = 100 − min(100, (yaw_deg/25)×50 + (pitch_deg/20)×50)` (0° → 100; ~25° yaw or ~20° pitch → 50).  
- **Gaze:** `gaze_bonus = (1 − gaze_normalized)×15` with `gaze_normalized = min(1, gaze_dist/(face_scale×0.45))`.  
- **Sustained bonus:** Consecutive frames above threshold (50) add up to +28; consistency of recent average adds up to +8.  
- **Final = base_score + sustained_bonus + consistency_bonus**, clipped 0–100.

**g1_pupil_dilation**  
- Proxy: **eye_area / baseline_eye_area**. Median of recent (blinks excluded).  
- Bands: r ≥ 1.12 → 68–98; 1.06–1.12 → 52–68; 1.02–1.06 → 38–52; 0.98–1.02 → 28–46; 0.92–0.98 → 15–28; else lower.  
- **Debiasing:** When brow is raised above baseline, score is pulled toward 45 to avoid double-counting with eyebrow flash.

**g1_eyebrow_flash**  
- Baseline = median brow height (first half of buffer); threshold = face_scale×0.028.  
- **Raise + return** required: `raised = max_recent > baseline + threshold`, `returning = cur < max_recent − 0.5×threshold`.  
- If both: score 62 + min(36, magnitude×110). If only raised: 45 + min(35, magnitude×65). Small sustained raise: 42 + ramp. Else 32.

**g2_stillness**  
- Movement from buffer (landmark/pose variance). **mov_normalized** ≈ variance of motion.  
- Piecewise: very low movement → 38–54; low → 24–38; medium → 18–24; high movement → lower (e.g. max(18, 38 − (mov−0.008)×1200)).

**g2_gaze_shift_frequency**  
- Rate of yaw/pitch change over a short window. **shift_rate = mean(|Δyaw|) + mean(|Δpitch|)**.  
- Low shift (e.g. 0.5–1.5) maps to ~48–52; higher shift increases score (processing/System 2).

**g3_contempt**  
- Combines lip asymmetry, nose crinkle proxy, and brow/mouth cues. Multiple sub-scores (e.g. lip corner asymmetry, nose–mouth distance) with thresholds; max of condition-based sub-scores, then clipped 0–100.

**g4_smile_transition**  
- Uses buffer history of Duchenne and mouth/relax cues. Score high when transitioning from lower to higher Duchenne (genuine smile emerging), with **g1_duchenne** passed in for current frame.

**g4_fixed_gaze**  
- Low gaze variance over recent frames → high score (focused on camera/speaker).

All other signifiers follow the same idea: geometric or temporal measures from `cur`/`buf`/`lm` mapped via thresholds and piecewise linear (or similar) functions to 0–100. See `helpers.py` methods `_g1_*`, `_g2_*`, `_g3_*`, `_g4_*` for exact bands and coefficients.

---

## 4. Group means and composite engagement score (from signifiers)

The **ExpressionSignifierEngine** aggregates signifiers into **group means** g1, g2, g3, g4 and then into a single **composite engagement score** used internally (e.g. for spike detection and context). This is **separate** from the “basic metrics + weighted score” in §1.

### 4.1 Group means

- **Weights:** From `_get_weights()` (e.g. signifier weights and **group** weights `[0.35, 0.15, 0.35, 0.15]` for G1–G4).
- **Per group:** Weighted mean of that group’s signifier scores. G3 is stored as **resistance** (high raw = high resistance), but for the composite engagement score we use **g3 = 100 − g3_raw** so that **high g3 = low resistance = good**.

Formally (with optional signifier weights `sw` and group keys as in code):

- `g1 = weighted_mean(g1_keys)`, `g2 = weighted_mean(g2_keys)`, `g3_raw = weighted_mean(g3_keys)`, `g4 = weighted_mean(g4_keys)`
- **g3 = 100 − g3_raw**

### 4.2 Composite engagement score (signifier-based)

- **Raw:** `composite_raw = gw[0]×g1 + gw[1]×g2 + gw[2]×g3 + gw[3]×g4` (default gw = [0.35, 0.15, 0.35, 0.15]).
- **Bonus:** If `(g1 + g4)/2 > 62`, then `composite_raw = min(100, composite_raw + 8)`.
- **Penalty:** If `g3_raw > 38` (i.e. resistance above 38), then `composite_raw = max(0, composite_raw − 10)`.
- **Final:** `score = clip(composite_raw, 0, 100)`.

So: interest (G1) and decision-ready (G4) are rewarded when jointly high; resistance (G3_raw) is penalized.

---

## 5. Composite metrics (multimodal and condition-based)

**compute_composite_metrics()** in `helpers.py` produces the **0–100 composite metrics** (e.g. verbal_nonverbal_alignment, cognitive_load_multimodal, rapport_engagement) used in the dashboard and for coaching. Inputs: **group_means** (g1–g4), **signifier_scores**, **speech_tags**, optional **acoustic_tags** and **acoustic_negative_strength**, and optional **composite_weights** `cw`.

### 5.1 Speech strength (0–1)

- ** _category_strength(tags, categories, window_sec=12)**  
  - Matches tags whose `category` is in `categories` and `time ≥ now − 12`.  
  - `recency = mean over matches of (1 − (now−t)/12)`  
  - `count_norm = min(1, len(matches)/3)`  
  - `base = min(1, 0.5×recency + 0.5×count_norm)`  
  - If any match has `discourse_boost`: `base += 0.1`.  
  - Return `min(1, base)`.

Used as: **commit_interest**, **confusion_concern**, **objection_concern**, **interest_realization**, **interest_confirmation**, **timeline_budget**, **skepticism_speech**, **enthusiasm_speech**, **hesitation_speech**, **authority_speech** (each with its own category list).

### 5.2 Acoustic boosts (0–1 or scaled)

- Binary (or fixed) contributions from **acoustic_tags**: e.g. **has_uncertainty** (0.25), **has_tension** (0.25), **has_disengagement** (0.25), **has_roughness** (0.2), **has_falling** (0.15), **has_arousal_high** (0.15), **has_monotone** (0.2). Used additively in the formulas below.

### 5.3 Condition-based facial composites

** _condition_composite(scores, conditions, base_when_met, base_when_not)**  
- Each condition is `(key, min_inclusive, max_inclusive)`; `None` means no bound.  
- **Fulfillment** = fraction of conditions met (each condition: value in range or ≥ min or ≤ max).  
- **Score = base_when_not + fulfillment × (base_when_met − base_when_not)** (0–100).

Examples (all use `sig = signifier_scores`):

| Composite | Conditions (key, min, max) | base_met | base_not |
|-----------|----------------------------|----------|----------|
| topic_interest_facial | (g1_forward_lean, 55, 90), (g1_eye_contact, 70, None) | 80 | 25 |
| active_listening_facial | (g1_rhythmic_nodding, 45, None), (g1_parted_lips, 45, None), (g1_eye_contact, 60, None) | 78 | 20 |
| agreement_signals_facial | (g1_rhythmic_nodding, 50, None), (g1_duchenne, 55, None), (g1_eye_contact, 65, None) | 82 | 22 |
| closing_window_facial | (g4_smile_transition, 58, None), (g4_fixed_gaze, 60, None), (g4_mouth_relax, 52, None) | 85 | 25 |
| receptivity_facial | (g1_parted_lips, 48, None), (g1_softened_forehead, 55, None), (g1_duchenne, 52, None) | 76 | 22 |
| evaluating_thinking_facial | (g2_thinking_brow, 50, None), (g2_lip_pucker, 40, None), (g2_look_up_lr, 45, None) | 72 | 25 |
| cognitive_processing_facial | (g2_stillness, 50, None), (g2_gaze_shift_frequency, 48, None) | 70 | 20 |
| resistance_cluster_facial | max of two _condition_composite(… contempt/gaze_aversion …), (… lip_compression/gaze_aversion …) | 75/72 | 15 |
| withdrawal_facial | (g3_lip_compression, 45, None), (g3_gaze_aversion, 40, None), (g3_brow_lower_sustained, 45, None) | 78 | 15 |
| disagreement_facial | (g3_head_shake, 50, None), (g3_contempt, 45, None) | 80 | 18 |
| passive_listening_facial | (g1_eye_contact, None, 45), (g3_no_nod, 45, None), (g2_stillness, 50, None) | 75 | 15 |
| trust_openness_facial | (g1_facial_symmetry, 65, None), (g1_softened_forehead, 55, None), (g1_eye_contact, 65, None) | 72 | 25 |
| curious_engaged_facial | (g1_brow_raise_sustained, 50, None), (g1_eye_contact, 65, None), (g1_mouth_open_receptive, 50, None) | 74 | 22 |

All results are clipped to [0, 100].

### 5.4 Multimodal composite formulas

- **verbal_nonverbal_alignment**  
  - `face_positive = (g4×0.5 + g1×0.5)/100`  
  - `align_raw = (commit_interest×w_speech + face_positive×w_face)×100` (default w_speech=0.6, w_face=0.4).  
  - If topic_interest_facial ≥ 65: `align_raw = min(100, align_raw + 8)`.  
  - Output: clip(align_raw, 0, 100).

- **cognitive_load_multimodal**  
  - `load_raw = (g2/100×w_g2 + confusion_concern×w_speech)×100` (default w_g2=0.55, w_speech=0.45).  
  - `load_raw = min(100, load_raw + has_uncertainty×100)`.  
  - Output: clip(load_raw, 0, 100).

- **rapport_engagement**  
  - `rapport_raw = (g1/100×w_g1 + interest_realization×w_speech + g3/100×w_g3)×100` (default 0.4, 0.35, 0.25).  
  - If active_listening_facial ≥ 60: `rapport_raw = min(100, rapport_raw + 6)`.  
  - Output: clip(rapport_raw, 0, 100).

- **skepticism_objection_strength**  
  - `g3_resistance = 100 − g3`.  
  - `skept_raw = (objection_concern×w_obj + (g3_resistance/100)×w_res)×100` (default 0.6, 0.4).  
  - `skept_raw = min(100, skept_raw + (has_tension + has_roughness)×100)`.  
  - Output: clip(skept_raw, 0, 100).

- **decision_readiness_multimodal**  
  - `ready_raw = (g4/100×w_g4 + commit_interest×w_ci)×100` (default 0.55, 0.45).  
  - If closing_window_facial ≥ 70: `ready_raw = min(100, ready_raw + 10)`.  
  - Output: clip(ready_raw, 0, 100).

- **opportunity_strength**  
  - `opp_raw = (decision_readiness/100×w_dr + verbal_nonverbal_alignment/100×w_vn)×100` (default 0.55, 0.45).  
  - Output: clip(opp_raw, 0, 100).

- **trust_rapport**  
  - `trust_raw = (g1/100×w_g1 + interest_realization×w_speech + g3/100×w_g3)×100` (default 0.5, 0.3, 0.2).  
  - If receptivity_facial ≥ 65: `trust_raw = min(100, trust_raw + 7)`.  
  - Output: clip(trust_raw, 0, 100).

- **disengagement_risk_multimodal**  
  - `g3_res = (100−g3)/100`, `no_commit = 1 − commit_interest`.  
  - `dis_raw = ((100−g1)/100×w_g1_low + no_commit×w_nocommit + g3_res×w_res)×100` (default 0.35, 0.35, 0.30).  
  - `dis_raw = min(100, dis_raw + has_disengagement×100)`.  
  - Output: clip(dis_raw, 0, 100).

- **confusion_multimodal**  
  - `conf_raw = (g2/100×0.45 + confusion_concern×0.40)×100 + has_uncertainty×25`.  
  - Output: clip(conf_raw, 0, 100).

- **tension_objection_multimodal**  
  - `tension_raw = ((100−g3)/100×0.45 + objection_concern×0.40)×100 + (has_tension + has_roughness)×25`.  
  - Output: clip(tension_raw, 0, 100).

- **loss_of_interest_multimodal**  
  - `loss_raw = ((100−g1)/100×0.45 + no_commit×0.40)×100 + has_disengagement×25 + acoustic_negative_strength×15`.  
  - Output: clip(loss_raw, 0, 100).

- **decision_plus_voice**  
  - `voice_boost = min(0.2, has_falling + has_arousal_high)`.  
  - `score = (decision_readiness_multimodal/100 + voice_boost)×100`.  
  - Output: clip(score, 0, 100).

- **psychological_safety_proxy**  
  - `low_tension = 1 − (has_tension + has_roughness)`.  
  - `safety_raw = (g3/100×0.45 + interest_confirmation×0.35 + low_tension×0.2)×100`.  
  - Output: clip(safety_raw, 0, 100).

- **urgency_sensitivity**  
  - `urgency_raw = (timeline_budget×0.4 + has_arousal_high×4×0.3 + g4/100×0.3)×100`.  
  - Output: clip(urgency_raw, 0, 100).

- **skepticism_strength**  
  - `skept2_raw = (skepticism_speech×0.5 + (100−g3)/100×0.3)×100`; then `min(100, skept2_raw + (has_tension+has_roughness)×50)`.  
  - Output: clip(skept2_raw, 0, 100).

- **enthusiasm_multimodal**  
  - `enth_raw = (enthusiasm_speech×0.4 + g1/100×0.4 + has_arousal_high×2×0.2)×100`.  
  - Output: clip(enth_raw, 0, 100).

- **hesitation_multimodal**  
  - `hes_raw = (hesitation_speech×0.4 + g2/100×0.4 + has_uncertainty×2×0.2)×100`.  
  - Output: clip(hes_raw, 0, 100).

- **authority_deferral**  
  - `auth_raw = (authority_speech×0.5 + g2/100×0.3 + has_monotone×2×0.2)×100`.  
  - Output: clip(auth_raw, 0, 100).

- **rapport_depth**  
  - `rap_raw = (g1/100×0.45 + interest_confirmation×0.35 + has_falling×2×0.2)×100`.  
  - Output: clip(rap_raw, 0, 100).

- **cognitive_overload_proxy**  
  - `still_norm = signifier_scores["g2_stillness"]/100`.  
  - `overload_raw = (g2/100×0.35 + confusion_concern×0.35 + has_uncertainty×2×0.15 + still_norm×0.15)×100`.  
  - Output: clip(overload_raw, 0, 100).

Default weights can be overridden via the **composite_weights** dict passed into **compute_composite_metrics** (see `cw.get("...", {})` in code).

---

## 6. Summary

| Layer | What it is | Where computed |
|-------|------------|----------------|
| **Overall score** | 0–100 engagement from basic metrics with weights + weakest-link | EngagementScorer.calculate_score (helpers.py) |
| **Basic metrics** | attention, eye_contact, facial_expressiveness, head_movement, symmetry, mouth_activity | engagement_detector._metrics_from_signifiers (from signifiers ± scorer hybrid) |
| **Core signifiers** | 40 G1–G4 metrics, 0–100 each | ExpressionSignifierEngine.get_all_scores (helpers.py) |
| **Group means** | g1, g2, g3_raw, g3=100−g3_raw, g4 | ExpressionSignifierEngine.get_group_means |
| **Signifier composite score** | Single 0–100 from g1–g4 + bonus/penalty | _composite_from_group_means, get_composite_score |
| **Composite metrics** | verbal_nonverbal_alignment, cognitive_load_multimodal, rapport_engagement, etc. | compute_composite_metrics (helpers.py) |

All metrics are **continuous 0–100** (or 0–1 for internal strengths). No binary hysteresis in the main pipelines; clamping and piecewise linear bands provide the score behavior.
