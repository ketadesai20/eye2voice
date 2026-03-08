| Model | Description | Test Acc | Best Val Acc | Best Val Loss | Best Epoch | Final Gap | Straight F1 | Down F1 | Left F1 | Right F1 | Up F1 | Notes |
|-------|-------------|----------|-------------|---------------|------------|-----------|-------------|---------|---------|----------|-------|-------|
| v16A | Nichol's CNN (no geo features) | 78% | 76.88% | — | 4 | 16.68% | 0.67 | 0.78 | 0.87 | 0.84 | 0.69 | Severe overfitting by epoch 7. |
| m1 | Added 7 geo features via MLP branch | 78% | ~76% | — | 2-3 | ~15% | 0.67 | 0.78 | 0.86 | 0.84 | 0.69 | Features work directionally but CNN overfits before MLP can learn. |
| m2 | Label smoothing (0.1) | 78% | 77.01% | 0.8208 | 5 | 14.59% | 0.67 | 0.78 | 0.86 | 0.84 | 0.72 | Bought 2-3 more epochs before overfitting. Mild positive. |
| m3 | Smaller FC (256→128) + dropout 0.7 | 77% | 76.02% | 0.8372 | 7 | 15.32% | 0.66 | 0.78 | 0.86 | 0.84 | 0.70 | Hurt accuracy. Model lost needed capacity. |
| m4 | CNN freezing after val loss plateau | 77% | 76.32% | 0.8355 | 4 | 13.17% | 0.66 | 0.79 | 0.86 | 0.84 | 0.65 | Phase 2 never improved over phase 1. |
| m5 | 4 classes + label smoothing + CNN freezing | 92% | 91.96% | 0.5595 | 4 | 7.16% | n/a | 0.92 | 0.94 | 0.92 | 0.75 | Dropping Straight was the breakthrough. |
| m6 | 4 classes + label smoothing, no CNN freezing | — | — | — | — | — | n/a | — | — | — | — | Ablation: does removing CNN freezing help in 4-class? |