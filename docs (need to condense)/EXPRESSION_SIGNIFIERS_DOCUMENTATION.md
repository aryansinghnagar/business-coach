# Expression Signifiers Documentation

## Overview

The Expression Signifier System analyzes facial landmarks to compute **30 distinct expression signifiers**, each scored from 0-100. These signifiers are organized into 4 groups that represent different aspects of engagement in business meetings.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Group 1: Interest & Engagement](#group-1-interest--engagement)
3. [Group 2: Cognitive Load](#group-2-cognitive-load)
4. [Group 3: Resistance](#group-3-resistance)
5. [Group 4: Decision-Ready](#group-4-decision-ready)
6. [Composite Score Calculation](#composite-score-calculation)
7. [Technical Details](#technical-details)
8. [Interpretation Guide](#interpretation-guide)

---

## System Architecture

### Core Components

- **ExpressionSignifierEngine**: Main class that processes facial landmarks
- **Temporal Buffer**: Stores recent frame data (default: 45 frames ≈ 1.5 seconds at 30fps)
- **Baseline Tracking**: Maintains adaptive baselines for eye area, EAR, and Z-depth
- **Size Invariance**: All metrics normalize by `face_scale` (inter-ocular distance)

### Workflow

1. **Frame Processing** (`update()`): Extracts features from landmarks and stores in buffer
2. **Score Calculation** (`get_all_scores()`): Computes all 30 signifier scores
3. **Composite Aggregation** (`get_composite_score()`): Combines groups into final engagement score

---

## Group 1: Interest & Engagement

**Purpose**: Measures positive engagement signals indicating interest and active participation.

**Default Weight**: 35% of composite score

### g1_duchenne: Genuine Smile Detection

**What it measures**: Duchenne smile (genuine smile with eye squinch/crow's feet)

**Calculation**:
- **Smile Component**: Measures mouth corner lift relative to face scale
  - Corner lift = (mid_y - average_corner_y)
  - Threshold: 3% of inter-ocular distance minimum
  - Intensity: 0-1 scale based on lift magnitude
- **Squinch Component**: EAR reduction relative to baseline
  - Optimal squinch: EAR at 85-95% of baseline (slight narrowing)
  - Too much squint (<85%) = less genuine
- **Combined Score**:
  - Both present: 50-100 (genuine Duchenne)
  - Smile only: 50-80 (social smile)
  - Squinch only: 45-55 (neutral)

**Score Interpretation**:
- **80-100**: Strong genuine smile, high engagement
- **60-80**: Moderate genuine smile or strong social smile
- **50-60**: Weak smile or neutral expression
- **0-50**: No smile, disengaged

**Business Context**: High scores indicate positive reception, agreement, or satisfaction with the conversation.

---

### g1_pupil_dilation: Eye Widening Proxy

**What it measures**: Eye area relative to baseline (proxy for pupil dilation/interest)

**Calculation**:
- Compares current eye area to temporal baseline
- Baseline updates only during non-blink frames (EMA: 95% old, 5% new)
- Ratio = current_area / baseline_area
- **Scoring**:
  - Ratio > 1.1: Dilated (60-100) - high interest
  - Ratio 1.0-1.1: Slightly wider (50-60) - moderate interest
  - Ratio 0.9-1.0: Normal (40-50) - baseline
  - Ratio < 0.9: Narrowed (0-40) - low interest
- Temporal smoothing: 70% current, 30% recent average

**Score Interpretation**:
- **70-100**: Eyes wide open, very attentive
- **50-70**: Normal to slightly dilated, engaged
- **30-50**: Normal baseline
- **0-30**: Eyes narrowed, disengaged or tired

**Business Context**: Widened eyes indicate interest, surprise, or active attention. Narrowed eyes suggest disengagement or fatigue.

---

### g1_eyebrow_flash: Rapid Eyebrow Movement

**What it measures**: Quick eyebrow raise-and-return pattern (surprise/acknowledgment)

**Calculation**:
- Tracks eyebrow height over time
- Detects pattern: raised → peak → returning
- **Flash Detection**:
  - Raised: Current height > baseline + 4% of face scale
  - Returning: Current height dropped from recent peak
- **Scoring**:
  - Strong flash (raised + returning): 60-95
  - Raised but not returning: 50-70
  - No flash: 50 (neutral)

**Score Interpretation**:
- **70-95**: Strong eyebrow flash, active acknowledgment
- **50-70**: Moderate eyebrow movement
- **<50**: No significant eyebrow activity

**Business Context**: Eyebrow flashes indicate recognition, surprise, or acknowledgment of information. Common during "aha" moments.

---

### g1_eye_contact: Gaze Direction Quality

**What it measures**: How directly the person is looking at the camera/speaker

**Calculation**:
- **Gaze Distance**: Distance from eye center to frame center (normalized by face scale)
- **Head Orientation Penalty**: 
  - Yaw deviation: >30° = significant turn
  - Pitch deviation: >20° = looking up/down
- **Combined Score**: Gaze quality penalized by head orientation (up to 60% reduction)

**Score Interpretation**:
- **80-100**: Strong eye contact, fully engaged
- **60-80**: Good eye contact, attentive
- **40-60**: Moderate eye contact, may be distracted
- **0-40**: Poor eye contact, likely multitasking

**Business Context**: Direct eye contact indicates focus and active listening. Averted gaze suggests distraction or disengagement.

---

### g1_head_tilt: Head Orientation

**What it measures**: Head roll angle (sideways tilt)

**Calculation**:
- **Optimal Range**: 5-15° tilt (shows interest, listening)
  - Peak at ~10°: 70-95 score
- **Very Slight** (0-5°): 55-70 (neutral-positive)
- **Moderate** (15-25°): 40-60 (less engaged)
- **Extreme** (>25°): 10-40 (disengaged)

**Score Interpretation**:
- **80-95**: Optimal tilt, showing interest
- **60-80**: Good tilt, engaged
- **40-60**: Moderate tilt, neutral
- **<40**: Extreme tilt or no tilt, disengaged

**Business Context**: Slight head tilt indicates active listening and interest. Extreme tilt suggests disengagement or discomfort.

---

### g1_forward_lean: Proximity to Camera

**What it measures**: Face Z-depth reduction (leaning forward)

**Calculation**:
- Compares current Z-depth to baseline
- Forward lean: Z < 97% of baseline
- Score: 50 + (lean_amount × 150), capped at 100

**Score Interpretation**:
- **70-100**: Strong forward lean, very engaged
- **50-70**: Moderate forward lean, interested
- **<50**: No lean or leaning back, neutral/disengaged

**Business Context**: Leaning forward indicates interest and active participation. Leaning back suggests passivity or disengagement.

---

### g1_facial_symmetry: Facial Symmetry

**What it measures**: Bilateral symmetry of facial features (left vs. right side of face)

**Calculation**:
- Checks symmetry of eyes, eyebrows, mouth, and nose alignment
- Mirrors right side features across face center (vertical midline) and compares to left side
- Measures both horizontal and vertical symmetry
- Normalizes errors by face scale (inter-ocular distance) for size invariance
- Uses weighted combination: eyes (40%), mouth (30%), brows (20%), nose (10%)
- Horizontal symmetry weighted 75%, vertical symmetry 25% (horizontal is more perceptually important)
- Accounts for natural facial asymmetry in realistic scoring curve

**Scoring Curve** (error as percentage of inter-ocular distance):
- **< 3% error**: Excellent symmetry (90-100 score)
- **3-6% error**: Good symmetry (70-90 score)
- **6-10% error**: Moderate symmetry (50-70 score)
- **10-15% error**: Poor symmetry (30-50 score)
- **> 15% error**: Very poor symmetry (0-30 score)

**Score Interpretation**:
- **90-100**: Excellent symmetry, very balanced facial expression indicating high focus
- **70-90**: Good symmetry, balanced expression indicating engaged attention
- **50-70**: Moderate symmetry, some asymmetry but generally attentive
- **30-50**: Poor symmetry, noticeable asymmetry indicating possible distraction or discomfort
- **0-30**: Very poor symmetry, significant asymmetry indicating distraction, discomfort, or lack of focus

**Business Context**: Symmetric facial expressions indicate focused engagement and attentiveness. Asymmetric expressions may indicate distraction, discomfort, or lack of engagement. This metric measures the internal symmetry of a single person's face along the vertical center line, not interpersonal mirroring behavior. The scoring accounts for natural human facial asymmetry (most faces have 3-8% natural asymmetry).

---

### g1_rhythmic_nodding: Vertical Head Oscillations

**What it measures**: Vertical head movement pattern (pitch oscillations) indicating agreement

**Calculation**:
- Analyzes pitch variation over 20-30 frames
- Detects zero-crossings in pitch derivative (oscillations)
- **Optimal Pattern**: 2-4 oscillations with 3-15° pitch range
- **Scoring**:
  - Strong pattern (2-4 oscillations, good amplitude): 75-95
  - Moderate pattern: 60-75
  - No pattern or too little movement: 30-45

**Score Interpretation**:
- **75-95**: Strong rhythmic nodding, active agreement
- **60-75**: Moderate nodding, agreement
- **40-60**: Minimal or irregular nodding
- **<40**: No nodding, passive

**Business Context**: Rhythmic nodding indicates agreement, understanding, or encouragement. Absence suggests disagreement or disengagement.

---

### g1_parted_lips: Mouth Openness

**What it measures**: Mouth Aspect Ratio (MAR) - vertical/horizontal mouth dimensions

**Calculation**:
- MAR = mouth_height / mouth_width
- **Optimal Range**: 0.12-0.35 (speaking readiness)
  - 0.12-0.35: 50-95 (increasing with MAR)
  - 0.08-0.12: 35-50 (slightly open)
  - 0.35-0.5: 65-85 (very open)
  - <0.08 or >0.5: 40 (closed or too open)

**Score Interpretation**:
- **70-95**: Mouth open, ready to speak or actively engaged
- **50-70**: Moderate openness, engaged
- **35-50**: Slightly open, neutral
- **<35**: Closed mouth, passive

**Business Context**: Parted lips indicate readiness to speak, active listening, or engagement. Closed lips suggest passivity.

---

### g1_softened_forehead: Low Brow Tension

**What it measures**: Eyebrow variance (low = relaxed, high = tense/furrowed)

**Calculation**:
- Calculates variance of eyebrow landmark positions
- Normalizes by face scale squared (variance scales with area)
- **Scoring**:
  - Very low variance (<0.002): 75 (relaxed)
  - Low variance (<0.008): 65 (mostly relaxed)
  - Higher variance: 35-70 (tense, decreasing score)

**Score Interpretation**:
- **70-75**: Very relaxed forehead, comfortable
- **60-70**: Relaxed, engaged
- **40-60**: Moderate tension, neutral
- **<40**: Tense forehead, possible stress

**Business Context**: Softened forehead indicates comfort and engagement. Tense forehead suggests stress or concentration.

---

## Group 2: Cognitive Load

**Purpose**: Measures cognitive processing indicators (thinking, evaluating, processing information).

**Default Weight**: 15% of composite score

### g2_look_up_lr: Memory Access Pattern

**What it measures**: Looking up and left/right (accessing memory, thinking)

**Calculation**:
- **Looking Up**: Pitch < -5° (nose up, eyes up)
- **Looking Left/Right**: Yaw > 10° (significant turn)
- **Scoring**:
  - Both up AND left/right: 50-90 (strong cognitive load)
  - Just up: 50-75 (moderate cognitive load)
  - Just left/right: 50 (minimal indicator)
  - Forward: 45 (no cognitive load signal)

**Score Interpretation**:
- **70-90**: Strong cognitive processing, accessing memory
- **50-70**: Moderate cognitive load, thinking
- **45-50**: Minimal cognitive load indicators
- **<45**: Looking forward, not processing

**Business Context**: Looking up and to the side indicates accessing memory or thinking. Common when evaluating proposals or recalling information.

---

### g2_lip_pucker: Thinking Expression

**What it measures**: Lip pucker (lips pursed, thinking/evaluating)

**Calculation**:
- High MAR (>0.25) with relatively narrow mouth width
- Normalizes mouth width by face scale
- **Scoring**:
  - Very high MAR (>0.35) + narrow mouth: 75
  - High MAR (>0.30) + narrow: 60-70
  - Moderate MAR (>0.25): 50-65
  - Normal: 40

**Score Interpretation**:
- **70-75**: Strong lip pucker, deep thinking
- **60-70**: Moderate pucker, evaluating
- **50-60**: Slight pucker, processing
- **<50**: Normal mouth, not thinking

**Business Context**: Lip pucker indicates evaluation, consideration, or decision-making. Common when processing complex information.

---

### g2_eye_squint: Concentration

**What it measures**: Eye narrowing (EAR reduction) indicating concentration

**Calculation**:
- EAR = Eye Aspect Ratio (height/width)
- **Scoring**:
  - EAR 0.10-0.18: 50-95 (increasing with narrowing)
  - EAR < 0.10: 85 (very concentrated)
  - EAR > 0.18: 35 (normal, not concentrating)

**Score Interpretation**:
- **80-95**: Strong squint, very concentrated
- **60-80**: Moderate squint, concentrating
- **50-60**: Slight squint, focused
- **<50**: Normal eyes, not concentrating

**Business Context**: Eye squint indicates concentration, focus, or effort to understand. Common when processing complex information.

---

### g2_thinking_brow: Asymmetric Eyebrow

**What it measures**: One eyebrow raised higher than the other (thinking/curiosity)

**Calculation**:
- Compares left and right eyebrow Y positions
- Normalizes difference by face scale
- **Scoring**:
  - Asymmetry 4-24% of face scale: 70 (optimal thinking brow)
  - Asymmetry 2-30%: 55 (moderate)
  - Symmetric: 45 (not thinking)

**Score Interpretation**:
- **65-70**: Strong thinking brow, active processing
- **55-65**: Moderate thinking brow, curious
- **45-55**: Slight asymmetry, neutral
- **<45**: Symmetric, not thinking

**Business Context**: Asymmetric eyebrows (one raised) indicate curiosity, questioning, or active thinking. Common during Q&A or evaluation.

---

### g2_stillness: Facial Movement Stability

**What it measures**: Low facial landmark variance (stillness = focus)

**Calculation**:
- Calculates variance of all facial landmarks
- Normalizes by face scale squared
- Compares recent variance to threshold
- **Scoring**:
  - Very low variance (<0.002): 75 (very still, focused)
  - Low variance (<0.006): 60 (still, attentive)
  - Higher variance: 45 (moving, less focused)

**Score Interpretation**:
- **70-75**: Very still, highly focused
- **60-70**: Still, attentive
- **45-60**: Some movement, moderate focus
- **<45**: Active movement, less focused

**Business Context**: Stillness indicates focus and concentration. Excessive movement suggests restlessness or distraction.

---

### g2_lowered_brow: Furrowed Brow

**What it measures**: Brow-to-eye distance (smaller = furrowed = concentration)

**Calculation**:
- Distance between eyebrow and eye centers
- Normalizes by face scale
- **Scoring**:
  - Strongly furrowed (<15% of face scale): 80
  - Moderately furrowed (15-20%): 65-75
  - Slightly furrowed (20-25%): 55-65
  - Normal (>25%): 45

**Score Interpretation**:
- **75-80**: Strongly furrowed, deep concentration
- **65-75**: Moderately furrowed, concentrating
- **55-65**: Slightly furrowed, focused
- **<55**: Normal brow, not concentrating

**Business Context**: Furrowed brow indicates concentration, problem-solving, or cognitive effort. In Group 2 context, this is positive (engagement through thinking).

---

## Group 3: Resistance

**Purpose**: Measures negative engagement signals (disagreement, discomfort, disengagement).

**Default Weight**: 35% of composite score (inverted: `100 - score`)

**Note**: High scores in Group 3 indicate HIGH resistance. The composite score inverts these (100 - score), so high resistance = low engagement.

### g3_contempt: Asymmetric Smirk

**What it measures**: One mouth corner raised higher than the other (contempt/disdain)

**Calculation**:
- Compares left and right mouth corner Y positions
- Normalizes asymmetry by face scale
- **Scoring**:
  - Asymmetry > 8% of face scale: 30-100 (increasing with asymmetry)
  - Symmetric: 20 (no contempt)

**Score Interpretation**:
- **70-100**: Strong contempt, high resistance
- **50-70**: Moderate contempt, disagreement
- **30-50**: Slight contempt, mild resistance
- **<30**: No contempt, neutral

**Business Context**: Contempt indicates disagreement, disdain, or skepticism. High scores suggest strong resistance to the message.

---

### g3_nose_crinkle: Disgust Expression

**What it measures**: Nose vertical extent reduction (crinkling = disgust)

**Calculation**:
- Compares current nose height to recent average (last 8 frames)
- **Scoring**:
  - Height < 88% of average: 55-90 (strong crinkle)
  - Height < 95% of average: 35 (moderate crinkle)
  - Normal: 12 (no crinkle)

**Score Interpretation**:
- **70-90**: Strong nose crinkle, high resistance
- **50-70**: Moderate crinkle, discomfort
- **35-50**: Slight crinkle, mild resistance
- **<35**: No crinkle, neutral

**Business Context**: Nose crinkle indicates disgust, distaste, or strong negative reaction. High scores suggest significant resistance.

---

### g3_lip_compression: Tight Lips

**What it measures**: Inner MAR (distance between upper/lower lip centers relative to mouth width)

**Calculation**:
- Uses inner MAR (more accurate than bbox MAR)
- **Scoring**:
  - MAR < 0.05: 88 (very tight, high resistance)
  - MAR < 0.08: 68-78 (tight)
  - MAR < 0.12: 40-60 (moderate)
  - MAR < 0.18: 20-40 (slightly tight)
  - Normal: 15

**Score Interpretation**:
- **80-88**: Very tight lips, high resistance
- **60-80**: Tight lips, disagreement
- **40-60**: Moderate compression, mild resistance
  - **<40**: Normal lips, no resistance

**Business Context**: Lip compression indicates disagreement, withholding, or resistance. High scores suggest the person is holding back or disagreeing.

---

### g3_eye_block: Prolonged Eye Closure

**What it measures**: Extended eye closure (blocking out information)

**Calculation**:
- Counts consecutive frames with EAR < 0.10
- **Scoring**:
  - ≥20 frames (≥0.67s): 90 (strong block)
  - 10-20 frames (0.33-0.67s): 50-88 (moderate block)
  - <10 frames: 8 (normal blink)

**Score Interpretation**:
- **80-90**: Prolonged closure, high resistance
- **60-80**: Extended closure, disengagement
- **50-60**: Moderate closure, discomfort
- **<50**: Normal blinks, no blocking

**Business Context**: Prolonged eye closure indicates blocking out information, discomfort, or disengagement. High scores suggest active resistance.

---

### g3_jaw_clench: Tension

**What it measures**: Tight jaw (low MAR) + corners down (frown)

**Calculation**:
- **Primary**: Inner MAR < 0.13 (tight lips/jaw)
- **Supporting**: Mouth corners below mouth center (frown)
- **Scoring**:
  - MAR < 0.07: 78-95 (very tight, +12 if corners down)
  - MAR < 0.10: 52-70 (tight, +8 if corners down)
  - MAR < 0.13: 28-46 (moderate, +7 if corners down)
  - Normal: 12

**Score Interpretation**:
- **80-95**: Strong jaw clench, high resistance
- **60-80**: Moderate clench, tension
- **40-60**: Slight clench, mild resistance
- **<40**: Normal jaw, no tension

**Business Context**: Jaw clench indicates tension, disagreement, or stress. High scores suggest significant resistance or discomfort.

---

### g3_rapid_blink: Excessive Blinking

**What it measures**: Blink frequency in 2-second window

**Calculation**:
- Counts blinks in last 2 seconds
- **Scoring**:
  - ≥5 blinks: 85 (very rapid, high resistance)
  - 3-4 blinks: 50-70 (rapid)
  - <3 blinks: 10 (normal)

**Score Interpretation**:
- **70-85**: Very rapid blinking, high stress/resistance
- **50-70**: Rapid blinking, stress
- **20-50**: Slightly elevated, mild stress
- **<20**: Normal blinking, no stress

**Business Context**: Rapid blinking indicates stress, anxiety, or discomfort. High scores suggest the person is under pressure or resistant.

---

### g3_gaze_aversion: Looking Away

**What it measures**: Head orientation away from camera (pitch + yaw combined)

**Calculation**:
- Combines pitch and yaw deviations: `sqrt(pitch² + yaw²)`
- **Scoring**:
  - Total deviation > 20°: 30-90 (strong aversion)
  - Total deviation > 10°: 20-30 (moderate aversion)
  - Total deviation > 5°: 15-20 (slight aversion)
  - Forward: 10 (no aversion)

**Score Interpretation**:
- **70-90**: Strong gaze aversion, high resistance
- **50-70**: Moderate aversion, disengagement
- **30-50**: Slight aversion, distraction
- **<30**: Looking forward, engaged

**Business Context**: Gaze aversion indicates disengagement, discomfort, or disagreement. High scores suggest the person is avoiding eye contact.

---

### g3_no_nod: Absence of Nodding

**What it measures**: Lack of vertical head movement (no agreement signals)

**Calculation**:
- Analyzes pitch variation over 30 frames
- Detects oscillations (zero-crossings)
- **Scoring**:
  - Very still (std < 1.0, range < 3°): 80 (no nodding)
  - Minimal movement (std < 1.5, range < 5°, <2 oscillations): 60-75
  - No oscillations: 50
  - Nodding detected: 15 (low resistance)

**Score Interpretation**:
- **70-80**: No nodding, high resistance
- **60-70**: Minimal nodding, passive
- **50-60**: Some movement but no clear nodding
- **<50**: Nodding detected, engaged

**Business Context**: Absence of nodding indicates disagreement, passivity, or disengagement. High scores suggest the person is not agreeing or encouraging.

---

### g3_narrowed_pupils: Eye Squint (Resistance Context)

**What it measures**: Eye narrowing (low EAR) in resistance context

**Calculation**:
- EAR < 0.14 indicates narrowed eyes
- **Scoring**:
  - EAR < 0.14: 50-90 (increasing with narrowing)
  - Normal EAR: 20

**Score Interpretation**:
- **70-90**: Strongly narrowed, high resistance
- **50-70**: Moderately narrowed, resistance
- **30-50**: Slightly narrowed, mild resistance
- **<30**: Normal eyes, no resistance

**Business Context**: Narrowed eyes in resistance context indicate skepticism, disagreement, or negative evaluation. Different from Group 2 squint (concentration).

---

## Group 4: Decision-Ready

**Purpose**: Measures readiness indicators (relaxation, focus, genuine engagement).

**Default Weight**: 15% of composite score

### g4_relaxed_exhale: Tension Release

**What it measures**: Decrease in facial tension + mouth opening (relaxation signal)

**Calculation**:
- Compares recent (last 3 frames) to previous (5-10 frames ago)
- **Tension Drop**: Variance decrease > 30%
- **Mouth Opening**: MAR increase > 5%
- **Scoring**:
  - Both present: 75-95 (strong relaxation)
  - Just tension drop: 60-75 (moderate relaxation)
  - Neither: 45 (no relaxation signal)

**Score Interpretation**:
- **80-95**: Strong relaxed exhale, very ready
- **70-80**: Moderate relaxation, ready
- **60-70**: Slight relaxation, somewhat ready
- **<60**: No relaxation, not ready

**Business Context**: Relaxed exhale indicates tension release and readiness. High scores suggest the person is comfortable and ready to proceed.

---

### g4_fixed_gaze: Stable Gaze Direction

**What it measures**: Low gaze movement variance (focused attention)

**Calculation**:
- Calculates standard deviation of gaze position over 10 frames
- Normalizes by face scale
- **Scoring**:
  - Std < 30% of face scale: 70 (very stable)
  - Std < 70% of face scale: 55 (moderately stable)
  - Higher std: 40 (unstable)

**Score Interpretation**:
- **65-70**: Very stable gaze, highly focused
- **55-65**: Stable gaze, focused
- **45-55**: Moderate stability, attentive
- **<45**: Unstable gaze, distracted

**Business Context**: Fixed gaze indicates sustained focus and attention. High scores suggest the person is fully engaged and ready.

---

### g4_smile_transition: Sustained Genuine Smile

**What it measures**: Sustained moderate MAR + Duchenne smile (social to genuine transition)

**Calculation**:
- Requires Duchenne score ≥ 50
- Analyzes MAR over last 20 frames
- **Sustained Pattern**: Mean MAR > 0.18, std < 0.04 (consistent)
- **Scoring**:
  - Sustained + Duchenne ≥ 60: 80 (strong transition)
  - Duchenne ≥ 55: 60 (moderate transition)
  - Duchenne 50-55: 50 (weak transition)
  - Duchenne < 50: 45 (no transition)

**Score Interpretation**:
- **75-80**: Strong smile transition, very ready
- **60-75**: Moderate transition, ready
- **50-60**: Weak transition, somewhat ready
- **<50**: No transition, not ready

**Business Context**: Smile transition indicates shift from social to genuine engagement. High scores suggest authentic positive engagement and readiness.

---

## Composite Score Calculation

The composite engagement score combines all 4 groups with weighted averaging:

```
composite = 0.35 × Group1 + 0.15 × Group2 + 0.35 × (100 - Group3) + 0.15 × Group4
```

**Key Points**:
- **Group 1** (Interest): 35% weight - positive engagement signals
- **Group 2** (Cognitive Load): 15% weight - thinking/processing (neutral-positive)
- **Group 3** (Resistance): 35% weight - **INVERTED** (100 - score) - negative signals
- **Group 4** (Decision-Ready): 15% weight - readiness indicators

**Why Group 3 is Inverted**:
- High resistance scores = low engagement
- Inversion: `100 - resistance_score` converts resistance to engagement
- Example: 80 resistance → 20 engagement contribution

**Final Score Range**: 0-100
- **85-100**: Very High Engagement
- **70-85**: High Engagement
- **45-70**: Medium Engagement
- **25-45**: Low Engagement
- **0-25**: Very Low Engagement

---

## Technical Details

### Size/Position Invariance

All metrics use `face_scale` (inter-ocular distance) for normalization:
- Pixel thresholds are relative to face size
- Works across different camera distances and face sizes
- Example: `threshold = face_scale * 0.04` (4% of inter-ocular distance)

### Temporal Baselines

**Eye Area Baseline**:
- Updates only during non-blink frames
- Exponential Moving Average (EMA): 95% old, 5% new
- Used for pupil dilation detection

**EAR Baseline**:
- Updates only during non-blink frames
- EMA: 95% old, 5% new
- Used for Duchenne squinch detection

**Z-Depth Baseline**:
- Updates continuously
- EMA: 95% old, 5% new
- Used for forward lean detection

### Blink Detection

- **Blink Threshold**: EAR < 0.16
- **Blink Duration**: 2-8 frames (0.07-0.27s at 30fps)
- **Rapid Blink Window**: 2 seconds
- Blink frames excluded from baseline updates

### Temporal Smoothing

Several metrics use temporal smoothing to reduce noise:
- **Pupil Dilation**: 70% current, 30% recent average
- **Nose Crinkle**: Compares to recent 8-frame average
- **Relaxed Exhale**: Compares recent vs. previous time windows

---

## Interpretation Guide

### High Engagement Pattern

**Group 1 (Interest)**: High scores across multiple signifiers
- Strong Duchenne smile (80+)
- Good eye contact (70+)
- Moderate head tilt (70+)
- Rhythmic nodding (75+)

**Group 2 (Cognitive Load)**: Moderate scores
- Some thinking indicators (50-70)
- Not too high (overthinking = stress)

**Group 3 (Resistance)**: Low scores
- Low contempt, compression, aversion (<30)
- Minimal resistance signals

**Group 4 (Decision-Ready)**: High scores
- Relaxed exhale (70+)
- Fixed gaze (65+)
- Smile transition (70+)

**Result**: Composite score 80-100 (Very High Engagement)

---

### Low Engagement Pattern

**Group 1 (Interest)**: Low scores
- No smile (<50)
- Poor eye contact (<40)
- No nodding (<40)

**Group 2 (Cognitive Load)**: Variable
- May be high (overthinking) or low (not processing)

**Group 3 (Resistance)**: High scores
- High contempt (70+)
- Lip compression (80+)
- Gaze aversion (70+)
- No nodding (80+)

**Group 4 (Decision-Ready)**: Low scores
- No relaxation (<50)
- Unstable gaze (<50)
- No smile transition (<50)

**Result**: Composite score 0-25 (Very Low Engagement)

---

### Cognitive Processing Pattern

**Group 1 (Interest)**: Moderate
- Some engagement but not overwhelming

**Group 2 (Cognitive Load)**: High scores
- Looking up/left-right (70+)
- Lip pucker (70+)
- Eye squint (80+)
- Thinking brow (70+)
- Furrowed brow (75+)

**Group 3 (Resistance)**: Low
- Not resisting, just thinking

**Group 4 (Decision-Ready)**: Moderate
- Processing, not yet ready

**Result**: Composite score 50-70 (Medium Engagement, but actively processing)

---

## Best Practices

1. **Context Matters**: Interpret scores in context of the conversation
2. **Temporal Patterns**: Look for trends over time, not single-frame scores
3. **Group Balance**: Consider all groups, not just composite score
4. **Baseline Establishment**: Allow 1-2 seconds for baselines to stabilize
5. **Face Quality**: Ensure good face detection quality for accurate scores

---

## Troubleshooting

### Low Scores Despite Engagement

- **Check face detection quality**: Poor landmarks = inaccurate scores
- **Verify baselines**: May need time to establish
- **Check face scale**: Very small/large faces may affect normalization
- **Review individual signifiers**: Identify which groups are low

### High Resistance Scores

- **Verify Group 3 signifiers**: Check which resistance signals are high
- **Context check**: Resistance may be appropriate (e.g., critical evaluation)
- **Temporal analysis**: Is resistance sustained or momentary?

### Inconsistent Scores

- **Temporal smoothing**: Some metrics smooth over time
- **Baseline updates**: Baselines adapt over time
- **Face detection stability**: Unstable detection causes fluctuations

---

## References

- **Code Location**: `utils/expression_signifiers.py`
- **Main Class**: `ExpressionSignifierEngine`
- **Related Documentation**: 
  - `ENGAGEMENT_DETECTION_DOCUMENTATION.md` - User guide
  - `ENGAGEMENT_SYSTEM_README.md` - Technical overview

---

*Last Updated: Based on latest expression signifier improvements*
