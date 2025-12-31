# Mol-vHeat Experiment Log

## 🏆 Best Results

| Dataset | Best Metric | Config | Status |
|---------|-------------|--------|--------|
| **ESOL** | **RMSE: 0.97** | LR=2e-5, 400ep | ✅ Beat baseline (1.15)! |
| BBBP | ROC-AUC: 0.60 | - | ❌ Needs work |

---

## ESOL Experiments

| Version | LR | Epochs | Test RMSE | Notes |
|---------|-----|--------|-----------|-------|
| v1 | 1e-4 | 100 | 2.18 | Too fast |
| v2 | 5e-5 | 200 | 1.37 | Good |
| v3 | 1e-5 | 300 | 1.48 | Too slow |
| **v4** | **2e-5** | **400** | **0.97** | 🏆 Best! |

---

## BBBP Experiments

| Version | Settings | ROC-AUC | Notes |
|---------|----------|---------|-------|
| v1 | Simple | 0.60 | Best so far |
| v2 | + regularization | 0.47 | Overfit |

---

## Log Folders
- `esol_ep400_lr2e-5_1231_1005/` - **Best ESOL (0.97)** 🏆
- `bbbp_20251231_084457/` - Best BBBP (0.60)
