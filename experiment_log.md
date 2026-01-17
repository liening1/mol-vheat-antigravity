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

---

## Experiment History

### Jan 17, 2026 - ESOL Comprehensive Evaluation

**Evaluation Run:** New HPC with fresh training (epochs=400, lr=2e-5)

| Metric | Value |
|--------|-------|
| Test RMSE | 0.993 |
| **5-Fold CV RMSE** | **0.704 ± 0.040** |
| MAE | 0.800 |
| R² | 0.79 |
| Pearson r | 0.891 |
| Spearman ρ | 0.870 |
| Inference | 16 ms/sample |
| Memory | 572 MB |

**CV Fold Results:** 0.730, 0.662, 0.741, 0.736, 0.650

**Key Insight:** CV RMSE (0.70) is competitive with GNN methods like TChemGNN (0.73)

**Files:**
- `periodic reports/evaluation_esol/evaluation_esol_20260117_205711.json`
- `periodic reports/evaluation_esol/cv_esol_5fold.json`

---

### Dec 31, 2025 - Initial ESOL Training

Best result achieved with epochs=400, lr=2e-5: **RMSE 0.97**
