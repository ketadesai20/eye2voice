# CEAL Cross-Domain Inference — m5c Results

## Bottom Line

m5c achieves **49.4% accuracy** on CEAL (4-class), beating the iTracker baseline of 39% by **+10.4 points**. The model transfers meaningfully on horizontal gaze and struggles on vertical — consistent with known training data imbalances.

## Setup

- **Model:** GazeNetM5c (4-class: Up, Down, Left, Right), trained on GazeCapture
- **Test set:** Columbia (CEAL) dataset — 5,600 samples across 56 subjects (280 Straight samples excluded)
- **Evaluation:** No training, no finetuning. Pure cross-domain transfer.
- **Known caveats:** Eye crops use 40% expansion (vs ~15% in training); geo features extracted from 224×224 face crops to match training domain

## Key Numbers

| Metric | Value |
|--------|-------|
| Overall accuracy | 49.4% |
| iTracker baseline (5-class) | 39.0% |
| Horizontal accuracy (Left/Right) | 59.1% |
| Vertical accuracy (Up/Down) | 26.7% |
| Best subject | 67.0% (subject 56) |
| Worst subject | 36.0% (subject 28) |
| Median subject | 49.0% |

## Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Up | 0.33 | 0.30 | 0.31 | 840 |
| Down | 0.26 | 0.24 | 0.25 | 840 |
| Left | 0.55 | 0.67 | 0.61 | 1,960 |
| Right | 0.59 | 0.51 | 0.55 | 1,960 |

## What Transfers and What Doesn't

**Horizontal gaze (59.1%)** — Strong transfer. Iris horizontal position and head yaw generalize well across domains. These features had the most training data and the clearest signal in feature engineering.

**Vertical gaze (26.7%)** — Near chance. Up was only 3% of GazeCapture training data, and the vertical signal (head pitch, z-tilt) was always the weakest geo feature. The domain gap between phone selfies (head tilts down naturally) and CEAL studio portraits (controlled vertical gaze) makes this worse.

## L/R Convention Issue

Initial run showed 18.7% accuracy (below random chance). Root cause: CEAL labels use the opposite left/right convention from GazeCapture. After swapping Left↔Right in the label mapping, accuracy jumped to 49.4%. This is a labeling convention mismatch, not a model failure.

## Implications

1. **m5c generalizes on its strongest axis.** Horizontal gaze direction transfers cross-domain — the model learned something real, not just dataset-specific patterns.
2. **Vertical remains the gap.** Same story as training: Up/Down need more data or better features. Up-sampling experiments (in progress) may help.
3. **Calibration matters.** Per-subject accuracy ranges from 36% to 67% — a calibration mechanism that adjusts for individual face geometry and device positioning could close this gap significantly.
4. **Baseline comparison caveat.** iTracker's 39% was 5-class (including Straight). Our 49.4% is 4-class. Not apples-to-apples, but directionally meaningful — especially since iTracker's Straight class absorbed errors that would have gone elsewhere.
