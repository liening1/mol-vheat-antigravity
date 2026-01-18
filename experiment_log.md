# Mol-vHeat Experiment Log

## 🏆 Results Summary

| Dataset | Metric | Official Result | GCN Baseline | Status |
|---------|--------|-----------------|--------------|--------|
| **ESOL** | RMSE ↓ | **0.76** (scaffold) | 0.89 | 🏆 Beats GCN! |
| Lipophilicity | RMSE ↓ | 1.12 | ~0.7 | ⚠️ OK |
| FreeSolv | RMSE ↓ | 1.76 | ~1.2 | ❌ Needs work |
| BBBP | ROC-AUC ↑ | 0.60 | 0.85 | ❌ Needs work |
| Tox21 | ROC-AUC ↑ | - | ~0.75 | ⏳ Pending |
| ClinTox | ROC-AUC ↑ | - | ~0.90 | ⏳ Pending |

---

## Best Configs

| Dataset | Best Config | Official Result |
|---------|-------------|-----------------|
| ESOL | `--epochs 400 --lr 2e-5` | **RMSE 0.76** (scaffold) 🏆 |
| Lipophilicity | `--epochs 200 --lr 2e-5` | RMSE 1.12 |
| FreeSolv | `--epochs 300 --lr 2e-5` | RMSE 1.76 |
| BBBP | `--epochs 100 --lr 5e-5` | AUC 0.60 |

---

## Key Findings
- 🏆 **ESOL**: Official scaffold split RMSE 0.76 beats GCN (0.89)!
- ⚠️ Lipophilicity: Reasonable but room for improvement
- ❌ FreeSolv: Small dataset (643) causes issues
- ❌ BBBP: Classification tasks need more work

---

## Experiment History

### Jan 17, 2026 - Official MoleculeNet Evaluation (Scaffold Split) 🏆

**Official scaffold split - directly comparable to published GNN benchmarks**

| Metric | Value |
|--------|-------|
| **Test RMSE** | **0.7577** |
| Test MAE | 0.6052 |
| Test R² | 0.8538 |
| Pearson r | 0.9261 |
| Train/Val/Test | 904/113/111 |

**Comparison:** Beats GCN (0.89), close to TChemGNN (0.73)

**File:** `periodic reports/evaluation_esol/official_esol_scaffold.json`

---

### Jan 17, 2026 - Comprehensive Evaluation (Random Split)

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

**Files:**
- `periodic reports/evaluation_esol/evaluation_esol_20260117_205711.json`
- `periodic reports/evaluation_esol/cv_esol_5fold.json`

---

### Dec 31, 2025 - Initial ESOL Training

Best result achieved with epochs=400, lr=2e-5: **RMSE 0.97**
