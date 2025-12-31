# Mol-vHeat Experiment Log

## Summary Table

| Exp | Dataset | Key Changes | Test Metric | Status |
|-----|---------|-------------|-------------|--------|
| 1 | ESOL | Baseline (LR=1e-4) | RMSE: 2.18 | ❌ |
| 2 | ESOL | LR=5e-5, warmup | **RMSE: 1.37** | ✅ |
| 3 | BBBP v1 | LR=5e-5, warmup | ROC-AUC: 0.60 | ❌ |
| 4 | BBBP v2 | + dropout=0.3, smoothing=0.1, wd=5e-4 | ROC-AUC: 0.47 | ❌❌ |

---

## Analysis

### ESOL (Solubility) ✅
- **Best Result**: RMSE 1.37 (Exp 2)
- Close to published baselines (~1.15)
- Lower LR + warmup helped significantly

### BBBP (Classification) ❌
- **Best Result**: ROC-AUC 0.60 (Exp 3, v1)
- **v2 got WORSE** (0.47) despite regularization
- **Diagnosis**: Too much regularization → underfitting

---

## Next Steps for BBBP

The current approach is struggling. Consider:

1. **Lower dropout**: Try `--dropout 0.1` (same as regression)
2. **Remove label smoothing**: `--label_smoothing 0`
3. **Higher learning rate**: Try `--lr 1e-4`
4. **Use pretrained backbone**: If vHeat has ImageNet weights
5. **Different approach**: Graph-based methods may work better for BBBP

---

## Files
- `logs/esol_20251231_083513/` - ESOL best (RMSE: 1.37)
- `logs/bbbp_20251231_084457/` - BBBP v1 (AUC: 0.60)
- `logs/bbbp_20251231_091546/` - BBBP v2 (AUC: 0.47)
