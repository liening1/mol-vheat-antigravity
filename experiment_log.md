# Mol-vHeat Experiment Log

This document records all experiments, hyperparameters, and results.

---

## Experiment 1: ESOL Baseline
**Date:** 2025-12-31  
**Dataset:** ESOL (Water Solubility)  
**Task:** Regression

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealing |
| Model | MolVHeat (vHeat backbone) |

### Results
| Metric | Value |
|--------|-------|
| **Test RMSE** | **2.1848** |

### Notes
- First baseline run
- RMSE higher than expected (target: ~1.0)
- Possible issues: learning rate, model size, data augmentation

---

## Experiment 2: ESOL Improved (Planned)
**Status:** Pending

### Planned Changes
- [ ] Lower learning rate: 5e-5
- [ ] More epochs: 200
- [ ] Add learning rate warmup
- [ ] Better image normalization

---

## Experiment 3: BBBP Classification (Planned)
**Status:** Pending

### Hyperparameters (Planned)
| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Loss | BCEWithLogitsLoss |
| Metric | ROC-AUC |

---

## Summary Table

| Exp | Dataset | Epochs | LR | Test Metric |
|-----|---------|--------|-----|-------------|
| 1 | ESOL | 100 | 1e-4 | RMSE: 2.18 |
| 2 | ESOL | 200 | 5e-5 | TBD |
| 3 | BBBP | 100 | 1e-4 | TBD |
