| Model | Description | Test Acc | Best Val Acc | Best Val Loss | Best Epoch | Final Gap | Straight F1 | Down F1 | Left F1 | Right F1 | Notes |
|-------|-------------|----------|-------------|---------------|------------|-----------|-------------|---------|---------|----------|-------|
| v16A | Baseline CNN (no geo features) | 78% | 76.88% | — | 4 | 16.68% | 0.67 | 0.78 | 0.87 | 0.84 | Nichol's baseline. Severe overfitting by epoch 7. |
| m1 | Added 7 geo features via MLP branch | 78% | ~76% | — | 2-3 | ~15% | 0.67 | 0.78 | 0.86 | 0.84 | Features work directionally but CNN overfits before MLP can learn. |
| m2 | Label smoothing (0.1) | 78% | 77.01% | 0.8208 | 5 | 14.59% | 0.67 | 0.78 | 0.86 | 0.84 | Bought 2-3 more epochs before overfitting. Mild positive. |
| m3 | Smaller FC (256→128) + dropout 0.7 | 77% | 76.02% | 0.8372 | 7 | 15.32% | 0.66 | 0.78 | 0.86 | 0.84 | Hurt accuracy. Model lost needed capacity. |
| m4 | CNN freezing after val loss plateau | 77% | 76.32% | 0.8355 | 4 | 13.17% | 0.66 | 0.79 | 0.86 | 0.84 | Phase 2 (frozen CNN) never improved over phase 1. Best checkpoint from epoch 4. |
| m5 | Drop Straight class (4 classes) | — | — | — | — | — | n/a | — | — | — | Tests if Straight is root cause of overfitting. |