# How We Detect and Score Engagement: A Plain-Language Guide

This guide explains **what** the system is looking for, **why** it matters in a meeting, and **how** we turn that into a number from 0 to 100. It’s written so that anyone—without a technical background—can follow the logic, with extra detail for readers who want a deeper understanding.

**For exact formulas and code references,** see [formulas.md](formulas.md).

---

## What does “0 to 100” mean?

Every metric is a **score from 0 to 100**. Think of it like a dimmer switch:

- **0–30** — The signal is weak or absent (e.g. no eye contact, very still face).
- **30–60** — Moderate (e.g. some attention, mixed signals).
- **60–100** — Strong (e.g. clear eye contact, clear smile, clear interest).

Scores are **continuous**, not pass/fail. A 45 is “a bit above the middle”; a 75 is “clearly high.” This lets the system reflect small changes and nuance.

---

## The big picture: your overall engagement score

**What it is:** One number that summarizes how “engaged” the person on camera appears—focused, responsive, and present rather than distracted or checked out.

**Why it matters:** In meetings, we care whether someone is really there. One very low signal (e.g. eyes constantly averted) can matter more than several okay ones, so the overall score is designed to reflect that.

**How we get the number:**

1. We first compute six **basic metrics** (attention, eye contact, facial expressiveness, head movement, symmetry, mouth activity). Each is 0–100.
2. We combine them with **weights**. Eye contact and attention count most (about 26% and 24%); the others share the rest. So the raw score is a weighted average: “how strong is each signal, with focus on eyes and attention?”
3. We then apply a **weakest-link** rule: if any one of the six is very low, we scale the score down. So one bad metric (e.g. no eye contact) can’t be hidden by the others. The math is gentle enough that a single weak metric doesn’t over-penalize, but it does pull the final number down.

**Result:** A single 0–100 engagement score that reflects both “average” strength of signals and “don’t ignore the worst one.”

---

## Basic metrics: the six building blocks

These are the six inputs that feed into the overall engagement score. For each we explain: what we mean by it, how we detect it, and how we turn that into 0–100.

---

### 1. Attention

**What we mean:** Is the person “here” and alert—eyes open, focused, not drowsy or looking away?

**How we detect it:**  
We use **eyes** as the main cue. When someone is paying attention, their eyes are typically open and steady; when they’re zoning out or looking down, eye openness and position change. We measure:

- **Eye openness** (from face landmarks): an “eye aspect ratio”—how open the eyes are. Very open ≈ attentive; very closed or squinted ≈ less so.
- **Balance between left and right eye:** If one eye is much more closed than the other (e.g. squint or wink), we treat that as less “full attention” and reduce the score a bit.

When we have a good view of the face, we also blend in a second estimate based on head orientation and gaze, so that “eyes open but head turned away” doesn’t get full marks.

**How we score it (simple):**  
We map eye openness to bands: “ideal” openness gives scores in the 70–100 range; too closed or too wide (e.g. surprise) gives lower scores. Then we subtract a small penalty if the two eyes are uneven. Final number is clamped between 0 and 100.

**Deeper:** The exact mapping is piecewise linear on “average EAR” (eye aspect ratio), with a peak band around 0.22–0.28. Asymmetry penalty is proportional to |left_ear − right_ear| / average_ear, capped so attention doesn’t go negative.

---

### 2. Eye contact

**What we mean:** Is the person facing the camera (and thus the speaker or audience)? Sustained facing counts more than a brief glance.

**How we detect it:**  
We use two things:

- **Head direction:** We estimate head yaw (turn left/right) and pitch (up/down). Facing the camera ≈ 0° turn; the more they turn away, the lower the score.
- **Where the eyes sit in the frame:** If the face is centered in the image, we add a small bonus; if it’s off to the side, we don’t.

We also look at **recent seconds** of video: if the person has been facing the camera for many frames in a row, we add a “sustained” bonus. So brief looks away don’t wipe out the score, but lasting eye contact is rewarded.

**How we score it (simple):**  
Start from head angle (0° → high score; ~25° yaw or ~20° pitch → about half score; 50°+ → low). Add a small bonus for face centered in frame. Then add up to +28 for sustained good contact over the last ~1 second, and a small bonus for consistency. Result is clamped 0–100.

**Deeper:** Head score = 100 − min(100, (yaw/25)×50 + (pitch/20)×50). Gaze bonus up to +15; sustained and consistency bonuses from consecutive frames above a threshold. See formulas.md for exact windows and thresholds.

---

### 3. Facial expressiveness

**What we mean:** Is the face “alive”—smiles, brow movement, open mouth, etc.—rather than flat or blank?

**How we detect it:**  
We don’t try to name every expression. We combine several **expression-related signals** the system already computes: genuine smile (Duchenne), parted lips, smile transitions, softened forehead, eyebrow flash, micro smile, eye widening, brow raise. The idea: if any of these are strong, the person is expressive.

**How we score it (simple):**  
We take the average and the maximum of those eight signals. Then we use **half the average plus half the maximum**. So one strong expression (e.g. a clear smile) can lift the score even if the rest are moderate—we want “at least one clear positive expression” to count. Final value is capped at 100.

**Deeper:** facial_expressiveness = 0.5×mean(expr_signals) + 0.5×max(expr_signals), then min(100, …). So the metric responds both to “overall level” and “peak” expressiveness.

---

### 4. Head movement

**What we mean:** Natural, engaged head motion—tilt, nodding, lean—rather than a completely frozen face (which can suggest disengagement or multitasking).

**How we detect it:**  
We combine:

- **Head tilt** (slight sideways tilt, which can signal listening).
- **Rhythmic nodding** (agreeing or “I’m following”).
- **Forward lean** (leaning toward the camera/speaker).
- **General motion:** We use “non-stillness”—i.e. how much the face is moving. So if the “stillness” signal is high, we treat that as *less* movement; if stillness is low, we treat that as more movement and feed it into this metric.

**How we score it (simple):**  
We average and max those four components (tilt, nod, lean, and “100 minus stillness”). Then we use half the average plus half the max—so one strong movement (e.g. clear nodding) can raise the score. Result is capped at 100.

**Deeper:** head_movement = min(100, 0.5×mean(head_components) + 0.5×max(head_components)), with head_components = [g1_head_tilt, g1_rhythmic_nodding, g1_forward_lean, 100 − g2_stillness].

---

### 5. Symmetry

**What we mean:** How balanced the face looks left-to-right. Strong asymmetry can indicate tension, skepticism, or a one-sided expression (e.g. smirk).

**How we detect it:**  
We compare the left half of the face (landmarks) to the mirror of the right half. The more they match, the higher the symmetry score.

**How we score it (simple):**  
We measure the average “error” between left and mirrored right. Small error → high score (85–100); larger error → lower. The exact bands are chosen so that normal small asymmetries don’t over-penalize. Result is 0–100.

**Deeper:** Symmetry is the signifier **g1_facial_symmetry** (no separate scorer path in the basic-metrics pipeline). That signifier is computed from landmark mirroring and error magnitude; see helpers.py.

---

### 6. Mouth activity

**What we mean:** Whether the mouth is moving or at least “available” for speech—slightly open, not tightly closed—which often goes with listening or about to speak.

**How we detect it:**  
We use the **parted lips** signal: how open or relaxed the mouth is (from mouth landmarks and, when available, mouth aspect ratio and openness indices).

**How we score it (simple):**  
mouth_activity is exactly the **parted lips** signifier score (0–100). So “mouth activity” here is “how much parted-lips signal we see,” which in practice reflects openness and readiness to speak or react.

**Deeper:** mouth_activity = g1_parted_lips. That signifier is driven by mouth geometry (e.g. MAR and openness) with piecewise bands; see _g1_parted_lips in helpers.py.

---

## Core signifiers: the fine-grained signals

Under the hood, the six basic metrics are built from **40 finer signals** called **signifiers**. Each signifier is a 0–100 score for one specific behavior. They are grouped into four families:

- **G1 — Interest & engagement:** Things that suggest the person is interested and present (smile, eye contact, nod, lean, etc.).
- **G2 — Cognitive load:** Things that suggest they’re thinking hard or processing (looking up, lip pucker, stillness, gaze shifting, etc.).
- **G3 — Resistance:** Things that suggest pushback or discomfort (contempt, lip compression, gaze aversion, head shake, etc.).
- **G4 — Decision-ready:** Things that suggest they’re ready to agree or close (relaxed mouth, steady gaze, smile settling in, etc.).

**How detection works (general):**  
For each signifier we use:

- **Geometry:** Positions and shapes of eyes, brows, mouth, head (from face landmarks).
- **Time:** Short clips of recent frames (a “buffer”) so we can see *patterns*—e.g. “brow went up then back down” (eyebrow flash) or “gaze stayed steady” (fixed gaze).

**How scoring works (general):**  
We turn one or a few numbers (e.g. “how much is the lip corner lifted?” or “how much has the head moved in the last second?”) into 0–100 using **bands**: e.g. “if this value is in this range, score is in that range.” So small changes in the face produce smooth changes in the score. We also avoid double-counting (e.g. when brow raise and “pupil dilation” both fire from the same gesture, we slightly reduce one).

**Examples in plain language:**

| Signifier | What we’re looking for | How we get the number |
|-----------|------------------------|------------------------|
| **Duchenne smile** | Genuine smile (cheek raise + lip corner pull). | We measure lip-corner lift and eye narrowing (EAR vs baseline). Both present → high score; synergy bonus when both are strong. |
| **Eye contact** | Head and gaze toward camera, sustained. | Head yaw/pitch near 0 → high; add bonus for face centered; add bonus for many consecutive frames with good contact. |
| **Eyebrow flash** | Quick “raise then return” of the brows (recognition, interest). | We need both “brow clearly above baseline” and “brow has come back down.” Only then do we give a high score; otherwise moderate or low. |
| **Stillness** | How little the face is moving. | We look at motion over recent frames. Very little motion → high stillness score; lots of motion → low stillness. |
| **Gaze shift frequency** | How often head/gaze direction changes. | We measure change in yaw/pitch over a short window. More shifting → higher “cognitive load” style score (processing). |
| **Contempt** | Lip asymmetry, nose/mouth cues associated with disdain. | We combine lip-corner asymmetry and other facial cues; if several cues pass thresholds, score goes up. |
| **Closing window** | Smile settling, gaze steady, mouth relaxed (ready to agree). | We check that “smile transition,” “fixed gaze,” and “mouth relax” are all above thresholds; the more conditions met, the higher the score. |

So: **detection** = “we measure X from the face (and sometimes recent frames)”; **scoring** = “we map X into 0–100 using bands and simple rules.” All 40 signifiers follow this idea; the technical doc has the exact bands and formulas.

---

## Composite metrics: combining face, speech, and voice

**Composites** are 0–100 scores that **combine**:

- **Face:** The signifiers and group summaries (G1–G4) above.
- **Speech:** Words we detect in the last ~12 seconds (e.g. “that’s interesting,” “I’m not sure,” “let’s do it”) and assign to categories like interest, confusion, objection, commitment.
- **Voice:** Optional acoustic tags (e.g. uncertainty, tension, disengagement) from tone and delivery.

So a composite answers a *higher-level* question like “how much do their words match their face?” or “how strong is objection or pushback?”

**How we combine (simple idea):**  
For each composite we define a **recipe**: which face signals, which speech categories, and (if used) which voice tags to mix, and with what weights. Some composites are “all face” (e.g. “are they in a closing-window pose?”); others blend face + speech or face + speech + voice. We then map that blend to 0–100, sometimes with small bonuses (e.g. “if they’re also leaning in, add a few points to alignment”).

**Examples in plain language:**

| Composite | What it’s for | How we get the number (simple) |
|-----------|----------------|--------------------------------|
| **Words match demeanor** | Do their words (e.g. “I’m in”) match a positive, engaged face? | We mix “commitment/interest” from speech (60%) with “positive face” from G1/G4 (40%). If they’re also “leaning in” (topic interest), we add a small bonus. |
| **Information overload** | Are they under cognitive load or confused? | We mix “cognitive load” face (G2) with “confusion/concern” from speech. If the voice sounds uncertain, we add a boost. |
| **Rapport with speaker** | Do they seem connected and receptive? | We mix interest face (G1), “interest/realization” from speech, and *low* resistance (G3). If “active listening” face is strong, we add a bonus. |
| **Objection or pushback** | Are they skeptical or objecting? | We mix “objection/concern” from speech with resistance from the face (G3). If voice has tension or roughness, we add. |
| **Ready to decide** | Are they ready to close or agree? | We mix “decision-ready” face (G4) with “commitment/interest” from speech. If “closing window” face is strong, we add a bonus. |
| **Attention slipping** | Risk of disengagement. | We mix *low* interest face (G1), *lack* of commitment language, and resistance (G3). If voice suggests disengagement, we add. |
| **Confusion or uncertainty** | Are they confused? | We mix “cognitive load” face (G2) with “confusion/concern” from speech and optional voice uncertainty. |
| **Closing opportunity** | Good moment to ask for commitment? | We mix “ready to decide” and “words match demeanor” so both verbal and nonverbal “readiness” count. |

**Condition-based composites (face only):**  
Some composites don’t use speech. They ask: “Are several face signals *all* present at once?” For example:

- **Active listening:** Nodding *and* parted lips *and* good eye contact above thresholds → high score; otherwise we scale down.
- **Agreement / buy-in signals:** Nodding *and* Duchenne smile *and* good eye contact → high score.
- **Closing window (ready to agree):** Smile transition *and* fixed gaze *and* mouth relax → high score.

The math: we compute “what fraction of the conditions are met?” and map that to a band (e.g. 20–85): all met → 85, none met → 20, half met → in between. So the score reflects “how many of the required signals are present.”

**Deeper:** Exact weights, thresholds, and bonus rules are in formulas.md (§5). Speech strength uses a 12-second window, recency, and count of matching phrases; face part uses group means (G1–G4) and/or individual signifiers.

---

## Putting it all together

1. **Face analysis** gives 40 signifier scores (0–100) and four group summaries (G1–G4).
2. **Basic metrics** are built from those signifiers (and optionally blended with landmark-based attention/eye contact). Those six numbers feed into the **overall engagement score** (weighted average + weakest-link).
3. **Composite metrics** combine face with speech (and sometimes voice) to answer questions like “words match demeanor?,” “objection strength?,” “ready to decide?,” “attention slipping?,” etc.

So: **detection** = “we measure something specific (geometry, time pattern, or words)”; **scoring** = “we map that into 0–100 so it’s easy to read and compare.” The logic is the same across the stack: define what “good” or “strong” looks like, then turn the raw signal into a 0–100 score with bands and simple rules.

For full formulas and code references, see [formulas.md](formulas.md).
