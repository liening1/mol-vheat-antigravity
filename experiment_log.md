# Mol-vHeat Experiment Log

## 🏆 Results Summary

| Dataset | Metric | Your Result | Baseline | Status |
|---------|--------|-------------|----------|--------|
| **ESOL** | RMSE ↓ | **0.97** | 1.15 | ✅ Beat baseline! |
| Lipophilicity | RMSE ↓ | 1.12 | ~0.7 | ⚠️ OK |
| FreeSolv | RMSE ↓ | 1.76 | ~1.2 | ❌ Needs work |
| BBBP | ROC-AUC ↑ | 0.60 | 0.85 | ❌ Needs work |
| Tox21 | ROC-AUC ↑ | - | ~0.75 | ⏳ Pending |
| ClinTox | ROC-AUC ↑ | - | ~0.90 | ⏳ Pending |

---

## Best Configs

| Dataset | Best Config | Best Result |
|---------|-------------|-------------|
| ESOL | `--epochs 400 --lr 2e-5` | **RMSE 0.97** 🏆 |
| Lipophilicity | `--epochs 200 --lr 2e-5` | RMSE 1.12 |
| FreeSolv | `--epochs 300 --lr 2e-5` | RMSE 1.76 |
| BBBP | `--epochs 100 --lr 5e-5` | AUC 0.60 |

---

## Key Findings
- ✅ ESOL: vHeat works well for water solubility prediction
- ⚠️ Lipophilicity: Reasonable but room for improvement
- ❌ FreeSolv: Small dataset (643) causes issues
- ❌ BBBP: Classification tasks need more work
