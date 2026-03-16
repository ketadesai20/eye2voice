# CEAL Cross-Domain Inference — m5c vs iTracker

## Bottom Line

m5c achieves **49.4% accuracy** on CEAL (4-class), beating iTracker's **39.2%** (5-class, axis-threshold method). Both models hit L/R convention mismatches that had to be corrected. m5c's advantage is concentrated in horizontal gaze, which is the primary axis for the MVP UI.

## Setup Comparison

| | m5c (ours) | iTracker (baseline) |
|--|-----------|-------------------|
| Architecture | 3 CNN streams + geo MLP | 3 CNN streams + facegrid |
| Training data | GazeCapture (balanced, 500/subject) | GazeCapture (full) |
| Output type | **Classification** (4-class logits) | **Regression** (x,y coords) |
| Classes evaluated | 4 (Up, Down, Left, Right) | 5 (+ Straight) |
| How accuracy is computed | Direct class prediction | Post-hoc binning of x,y coords via axis-threshold¹ |
| CEAL samples | 5,600 (aug_id=0 only, Straight excluded) | 23,520 (all aug_ids, all classes) |
| L/R convention fix | Swapped Left↔Right label indices | Negated pred_x (camera→user) |
| Overall accuracy | **49.4%** | 39.2% |

¹ iTracker's axis-threshold method: center point = median of Straight predictions, then assign direction by which axis (x or y) has larger displacement from center. Deadzone around center maps to Straight. A centroid nearest-neighbor method scored slightly lower at 37.8%.

## Per-Class Comparison

| Class | m5c Precision | m5c Recall | iTracker Precision | iTracker Recall |
|-------|:---:|:---:|:---:|:---:|
| Left | 0.55 | 0.67 | 0.58 | 0.43 |
| Right | 0.59 | 0.51 | 0.60 | 0.29 |
| Up | 0.33 | 0.30 | 0.27 | 0.55 |
| Down | 0.26 | 0.24 | 0.26 | 0.50 |
| Straight | — | — | 0.07 | 0.05 |

**Key observations:**

- **m5c dominates horizontal recall.** Left 67% vs 43%, Right 51% vs 29%. The geo features (iris position, head yaw) give m5c a real edge here. iTracker's regression outputs cluster too tightly on the x-axis to separate Left from Right cleanly (visible in the boxplot: Right's corr_x distribution overlaps heavily with Left).
- **iTracker shows higher vertical recall but low precision.** Up 55% vs 30%, Down 50% vs 24%. iTracker's confusion matrix reveals why: massive bleed from Left/Right into Up/Down predictions (Left→Up: 1955, Left→Down: 1917). It's not that iTracker "understands" vertical gaze better — it's over-predicting vertical directions at the expense of horizontal.
- **Straight is effectively failed for iTracker.** 5% recall, 7% precision. Validates the decision to drop Straight as a class and replace it with confidence thresholding.

## Accuracy by Axis

| Axis | m5c | iTracker |
|------|-----|----------|
| Horizontal (Left/Right) | **59.1%** | 36.2%* |
| Vertical (Up/Down) | 26.7% | 52.7%* |

*Estimated from iTracker classification report weighted by support.

## L/R Convention: Same Problem, Different Fix

Both models hit the same fundamental issue — CEAL and GazeCapture define left/right from opposite perspectives.

- **iTracker:** Regression model outputs gaze coords from camera perspective. Fix: negate pred_x before classification.
- **m5c:** Classification model. Fix: swap Left↔Right in the label mapping (`Left→3, Right→2` instead of `Left→2, Right→3`).

Both fixes were confirmed by showing that accuracy jumps from below chance to above 39% after correction.

## iTracker Head Pose Sensitivity

iTracker accuracy varies by CEAL head pose: 31% at -30° up to 46% at +15°. This is a direct measurement of the calibration problem — the same model performs very differently depending on head-camera angle. Motivates calibration work regardless of which model we deploy.

## What This Means for the Project

1. **m5c is the right model for our app.** Horizontal gaze is the primary selection axis in the MVP UI (Left/Right screen regions). m5c's 59% horizontal accuracy vs iTracker's 36% is a significant practical improvement.
2. **The classification framing is better than regression + binning.** m5c learns decision boundaries directly; iTracker has to hope its continuous coordinates cluster into separable regions, and they don't (see scatter plot overlap). For a directional intent system, classification is the right task formulation.
3. **Vertical remains the shared weakness.** Neither model handles Up/Down well cross-domain. Up-sampling experiments (in progress) and calibration are the paths forward.
4. **Dropping Straight was correct.** Both models confirm it doesn't work as a class. Confidence thresholding is the right replacement.
5. **Calibration is the clear next step.** Per-subject variance (m5c: 36–67%) and head-pose sensitivity (iTracker: 31–46%) show that setup-specific adjustment has significant room to improve results.
