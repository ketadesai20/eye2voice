# Calibration Experiments — Implementation Plans

## The Problem

m5c works well when the phone is held at the angle/distance it was trained on (GazeCapture: phone below eye level, handheld). Performance degrades when:
1. **Device angle changes** — tripod at chest height, phone on a stand, held at eye level
2. **Center drifts** — what "neutral" looks like varies by person + setup

These are two symptoms of the same root cause: the model learned one physical setup and gets confused by others.

## Testing Protocol (via Katya's app)

Each experiment is evaluated using the 24-trial testing app:
- **Metrics:** accuracy (correct/24), mean response time, per-direction accuracy
- **Conditions:** Each person runs the protocol under **at least 2 setups**:
  - Setup A: phone handheld at natural angle (should be the model's comfort zone)
  - Setup B: phone on tripod/stand at a different height or angle
  - Setup C (optional): phone at eye level, held by another person
- **Baseline first:** run m5c with no calibration under all setups to quantify the actual gap before trying to close it

### Critical: Establish the Baseline Gap First

Before running any calibration experiment, we need to confirm the problem exists in the testing app. If m5c scores 24/24 across all setups and testers, there's nothing to fix.

**Baseline protocol:**
- At least 3 different people run the 24-trial test
- Each person runs under setups A and B (minimum)
- Log all sessions

**Expected:** accuracy drops in setup B vs setup A. If it doesn't, calibration work is deprioritized.

---

## Experiment 1: Geo-Delta (No Retraining)

### What It Is
Subtract the user's resting geo features from live geo features at inference time. The model sees *deltas from neutral* instead of absolute values. If the phone is angled 15° lower than training data, head_pitch is offset — subtracting the resting value removes that offset.

### Implementation
1. Add a "look at center" screen before the 24 trials begin
2. On that frame, extract the 7 geo features via MediaPipe (same pipeline as training)
3. Cache those as `geo_baseline`
4. At each inference step: `geo_input = geo_live - geo_baseline`
5. Feed `geo_input` to m5c as usual (no model changes)

### What Changes in the Model
Nothing. Same m5c checkpoint. Only the input preprocessing changes.

### Success Criteria
- **Clear win:** Setup B accuracy improves by ≥5 points AND setup A accuracy doesn't drop by more than 2 points
- **Neutral:** Setup B improves by <3 points, setup A unchanged. Signal is too weak to matter.
- **Failure:** Setup A accuracy drops by >3 points (subtracting the baseline corrupts the signal the model relies on)

### Effort
Low. 1-2 days. No retraining, no architecture changes. Just preprocessing.

### Key Risk
m5c was trained on *absolute* geo features, not deltas. Subtracting the baseline shifts the input distribution away from what the model learned. The geo MLP might not generalize to delta-space inputs. If the geo branch contributes a small fraction of the model's accuracy, the delta might not help much either.

---

## Experiment 2: Logit Offset (No Retraining)

### What It Is
At setup time, user looks Up/Down/Left/Right on command (4 calibration frames). Record the model's raw logits for each. The average logit vector across these 4 known-direction frames reveals the model's directional bias for this setup. Subtract that bias at inference time.

### Implementation
1. Add 4 calibration prompts before the 24 trials: "look Up," "look Down," "look Left," "look Right"
2. For each, record the raw logit vector (4 values)
3. Compute `logit_offset = mean(logits across 4 calibration frames) - expected_logits`
   - Where expected_logits for "look Left" would be something like [0, 0, 1, 0] (one-hot, or use the training set's average logit for that class)
   - Simpler version: `logit_offset = mean(all 4 calibration logit vectors)` — if the model is unbiased, this should be near uniform
4. At inference: `adjusted_logits = raw_logits - logit_offset`

### What Changes in the Model
Nothing. Same m5c checkpoint. Post-processing only.

### Success Criteria
- **Clear win:** Setup B accuracy improves by ≥5 points, setup A stays within 2 points
- **Neutral:** <3 point improvement. The bias is already small.
- **Failure:** Accuracy drops in both setups. The offset overcorrects.
- **Bonus metric:** Compare against Experiment 1. If logit offset ≥ geo-delta, it's simpler and better.

### Effort
Low-medium. 2-3 days. No retraining. Requires 4 extra calibration frames in the testing app, which Katya would need to add.

### Key Risk
Assumes the model's error is a constant directional bias across all inputs. If the error is input-dependent (e.g., Left predictions are fine but Up predictions are shifted), a single global offset won't capture it.

---

## Experiment 3: Siamese Conditioning Branch (Retraining Required)

### What It Is
Add a new branch to the model architecture that takes a calibration image (Straight frame from the same subject) and produces an embedding. This embedding conditions the FC classifier so it adjusts predictions based on what "neutral" looks like for this person in this setup.

### Architecture Changes
```
Current m5c:
  left_eye  → eye_cnn  → 4608 ─┐
  right_eye → eye_cnn  → 4608 ─┤
  face      → face_cnn → 2304 ─┤
  geo (7)   → geo_mlp  →   64 ─┤
                                └→ FC(11584 → 512 → 256 → 4)

Proposed m6-calibration:
  left_eye  → eye_cnn  → 4608 ─┐
  right_eye → eye_cnn  → 4608 ─┤
  face      → face_cnn → 2304 ─┤
  geo (7)   → geo_mlp  →   64 ─┤
  cal_face  → cal_cnn  →  128 ─┤  ← NEW: calibration branch
                                └→ FC(11712 → 512 → 256 → 4)
```

Options for the calibration encoder:
- **Option A (simple):** small CNN (3-4 conv layers) → 128-dim embedding, concatenated into FC
- **Option B (FiLM):** embedding modulates the FC layers via learned scale+shift (more expressive, more complex)

Start with Option A.

### Training Changes
- For each training sample, randomly select a Straight frame from the *same subject* as the calibration input (637/640 subjects have Straight — 99.5% coverage)
- For the 3 subjects without Straight, use the frame with smallest gaze angle
- Data pipeline: WebDataset needs to be modified to sample pairs (training frame + calibration frame) per subject. This is the biggest implementation lift.

### Deployment
- User does "look at center" hello screen → one frame
- That frame goes through `cal_cnn` → embedding is cached
- All subsequent inferences concatenate the cached embedding into FC

### Success Criteria
- **Clear win:** Setup B accuracy improves by ≥8 points over uncalibrated m5c, AND per-subject variance on CEAL drops (std from 6.9% to <5%)
- **Worth the complexity:** Must beat Experiment 1 and Experiment 2 by ≥3 points to justify the architecture change
- **Neutral:** Accuracy improves by <5 points. The conditioning branch isn't learning anything useful beyond what the simpler approaches capture.
- **Failure:** Training doesn't converge, or accuracy regresses vs m5c. The calibration branch adds noise.

### Effort
High. 1-2 weeks. Architecture change, data pipeline modification, retraining on Colab.

### Key Risk
- Data pipeline complexity: sampling subject-matched pairs from WebDataset tars requires restructuring the dataloader (tars are sharded, not grouped by subject)
- The model might learn to ignore the calibration branch if the existing features are already sufficient for within-distribution accuracy. The calibration signal only matters on *out-of-distribution* setups, which are underrepresented in training.

---

## Experiment 4: CEAL as Calibration Input (Retraining Required)

### What It Is
Katya's original proposal. Use a CEAL image as the calibration reference during training. This would teach the model about canonical head poses from a controlled studio environment.

### How It Would Work
- Same architecture as Experiment 3 (calibration branch)
- During training: instead of a same-subject Straight frame, pair each GazeCapture sample with a CEAL image (different person, different camera, different setup)
- The CEAL image provides a "canonical pose reference" rather than a person-specific calibration

### Open Questions (Need Katya's Input)
- What signal should the model extract from the CEAL image? Person-independent pose information? If so, we'd want to pair by head pose angle (CEAL has controlled poses at -30°, -15°, 0°, +15°, +30°).
- How does the model learn to use a cross-person reference? The calibration branch sees a stranger's face — it can't learn person-specific adjustments.

### Success Criteria
- Same bar as Experiment 3 (must beat simpler approaches by ≥3 points)
- **Additional:** Does cross-domain generalization improve? (We lose CEAL as a test set but could hold out some CEAL subjects)

### Effort
High. Same architecture work as Experiment 3, plus CEAL integration into the training pipeline.

### Key Risk
- Train/deploy mismatch: training with CEAL references but deploying with the user's phone selfie. The calibration branch sees different input domains at train vs inference time.
- No shared signal between the CEAL face and the GazeCapture face — the model might learn to ignore the branch entirely.

### My Honest Take
I'd try this only if Experiment 3 shows the conditioning branch actually works. If same-subject conditioning can't improve things, cross-subject conditioning won't either. Think of Experiment 3 as the *upper bound* on what a calibration branch can achieve.

---

## Recommended Order

```
1. Baseline gap measurement (1 day)
   └─ Does the problem exist in the testing app?
   
2. Experiment 1: Geo-delta (1-2 days)
   └─ Cheapest test. Sets the floor for improvement.

3. Experiment 2: Logit offset (2-3 days)
   └─ If this matches geo-delta, stop here — the problem is a simple bias.
   └─ If it beats geo-delta, the model has a directional bias worth correcting.
   └─ If both fail, the problem is deeper than a simple offset.

4. Experiment 3: Siamese conditioning (1-2 weeks)
   └─ Only if 1+2 show the gap exists but aren't sufficient.
   └─ Must beat simpler approaches by ≥3 points.

5. Experiment 4: CEAL conditioning (1-2 weeks)
   └─ Only if Experiment 3 succeeds AND Katya's vision for CEAL is clarified.
```

## Decision Framework

After each experiment, ask:
- **Is the gap closed enough for the app?** If accuracy is >90% across setups after geo-delta, we're done.
- **Is the improvement worth the complexity?** A 3-point improvement from 2 weeks of architecture work is not worth it in a capstone timeline.
- **Does this change the user experience?** More calibration steps = worse UX. One "look at center" frame is fine. Four calibration prompts might be too much.
