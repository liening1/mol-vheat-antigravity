# Mol-vHeat Experiment Log

This document records all experiments, hyperparameters, and results.

---

## Summary Table

| Exp | Dataset | Epochs | LR | Warmup | Test Metric | Baseline | Status |
|-----|---------|--------|-----|--------|-------------|----------|--------|
| 1 | ESOL | 100 | 1e-4 | No | RMSE: 2.18 | 1.15 | ❌ Baseline |
| 2 | ESOL | 200 | 5e-5 | 5 epochs | **RMSE: 1.37** | 1.15 | ✅ Improved |
| 3 | BBBP | 100 | 5e-5 | 5 epochs | ROC-AUC: 0.60 | 0.85 | ❌ Needs work |

---

## Experiment 1: ESOL Baseline
**Date:** 2025-12-31 (HPC)  
**Status:** ❌ Completed - Below baseline

### Results
| Metric | Value |
|--------|-------|
| **Test RMSE** | **2.18** |

### Notes
- Default hyperparameters, no warmup
- Significant gap from target (~1.0)

---

## Experiment 2: ESOL Improved ✅
**Date:** 2025-12-31 (HPC run esol_20251231_083513)  
**Status:** ✅ Completed - Good improvement!

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch Size | 32 |
| Learning Rate | 5e-5 |
| Warmup Epochs | 5 |
| Weight Decay | 1e-4 |

### Results
| Metric | Value | Improvement |
|--------|-------|-------------|
| Best Val RMSE | 1.27 | - |
| **Test RMSE** | **1.37** | **37% better** than Exp 1 |

### Training Observations
- Loss started high (~5.5), dropped around epoch 97-100
- Model kept improving until epoch 178
- Final epochs showed convergence (LR near 0)

---

## Experiment 3: BBBP Classification
**Date:** 2025-12-31 (HPC run bbbp_20251231_084457)  
**Status:** ❌ Completed - Below baseline

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 32 |
| Learning Rate | 5e-5 |
| Warmup Epochs | 5 |

### Results
| Metric | Value | Target |
|--------|-------|--------|
| Best Val ROC-AUC | 0.74 | - |
| **Test ROC-AUC** | **0.60** | 0.85 |

### Analysis
- Val-Test gap suggests **overfitting**
- Best val at epoch 63 (0.74), but test only 0.60
- Need more regularization or data augmentation

---

## Next Steps

### For ESOL
- [x] Lower LR helped significantly
- [ ] Try even smaller LR (1e-5) for more epochs
- [ ] Try larger model depth

### For BBBP (Needs Improvement)
- [ ] Add dropout to MolVHeat head
- [ ] Increase weight decay
- [ ] Use label smoothing
- [ ] Try more aggressive augmentation
- [ ] Use scaffold split for better generalization

---

## Files
- `logs/esol_20251231_083513/` - ESOL v2 logs
- `logs/bbbp_20251231_084457/` - BBBP logs
- `archive/train_v1_baseline.py` - Original training script
