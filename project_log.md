# 📋 Project Action Log — Predictive Indiv (Student Tuition Debt Prediction)

> This log tracks all significant actions, decisions, and changes made throughout the project lifecycle.
> Entries sourced from: `logs/actions.log` + conversation history summaries.

---

## Session 1 — 2026-02-27 (~12:10 UTC)
**Source:** `logs/actions.log`

| Time (UTC) | Action |
|---|---|
| 12:10:55 | Created `logs/` folder and initial `actions.log` file |
| 12:14:28 | Reviewed `Predictive_Indiv.ipynb`. Identified redundant data loading steps (both `ucimlrepo` and direct CSV download from GitHub) and basic EDA steps |
| 12:19:09 | Refactored notebook — removed `pip install ucimlrepo` and `fetch_ucirepo` cells. Notebook now relies solely on direct CSV download from GitHub |
| 12:36:27 | Created composite `Financial_Distress` feature in notebook (`Debtor == 1 OR Tuition fees up to date == 0`) for **EDA exploration only**; dropped original `Target` (dropout) column. Note: `Financial_Distress` was NOT used as the model target — `Debtor` is the final target in `train_model.py` |
| 12:49:20 | Added EDA cells: dropped `Debtor` and `Tuition fees up to date` to prevent data leakage. Added visualizations for: Target Variable Distribution, Feature Correlations, Key Attribute Combinations (Scholarship holder, Age, Gender) |
| 12:55:37 | Fixed JSON double-escape newline syntax issue in newly added EDA cells that was causing Python execution errors |

---

## Session 2 — 2026-02-27 (~17:55 UTC)
**Source:** Conversation summary — "ML Project Stages 1-3"

### Goals Completed
- **Stage 1 — Problem Definition:** Refined the project problem statement (predicting student debtor risk)
- **Stage 2 — EDA Enhancement:** Expanded EDA with a comprehensive set of visualizations covering:
  - Data distributions
  - Class imbalance analysis
  - Missingness checks
  - Leakage risk assessment
  - Outlier detection
  - Data quality issues
- **Stage 3 — Data Preparation Pipeline:**
  - Established a robust preprocessing pipeline
  - Implemented **60/20/20 train/validation/test split**
  - All work done within `Predictive_Indiv.ipynb`

---

## Session 3 — 2026-02-27 (~17:56 UTC)
**Source:** Conversation summary — "Model Training Plan"

### Goals Completed
- Outlined the full ML pipeline plan:
  - Data acquisition → EDA → Model training → Fine-tuning → Results presentation
- Identified **Random Forest** as the primary model to build

---

## Session 4 — 2026-02-27 (~18:41 UTC)
**Source:** Current session

| Time (UTC) | Action |
|---|---|
| 18:41:52 | Created blank `train_model.py` file in project root (no code added yet) |
| 18:44:01 | Created `project_log.md` (this file) consolidating all previous logs and session summaries |
| 18:44:01 | Created `README.md` summarising project state |
| 18:52:37 | Added initial code to `train_model.py`: data loading from `dataset.csv`, risk strata engineering (`fin_pressure`, `acad_proxy`, `risk_strata`), stratified 80/20 train/test split using `StratifiedShuffleSplit`, cleanup of temp columns, and definition of `X_train`, `y_train`, `X_test`, `y_test` with `Debtor` as target |
| 18:55:55 | Updated `train_model.py`: corrected data source filename from `dataset.csv` to `data.csv` (GitHub source). No UCI retrieval code present — already removed in a prior session from the notebook |
| 18:58:08 | Updated `train_model.py`: replaced local `data.csv` path with direct raw GitHub URL — `https://raw.githubusercontent.com/EllieBril/Student_data_-UCI/main/data.csv` |

---

## Session 5 — 2026-02-28 (~08:19 UTC)
**Source:** Conversation summary — "Adjusting Data Split"

### Goals Completed

#### Data Split Revision
- Changed `train_model.py` from an 80/20 split to an explicit **60/20/20 (Train/Validation/Test)** split using two sequential `StratifiedShuffleSplit` calls with risk strata stratification
- Added **leakage checks**: target column not in features, no index overlap between splits, temp/leakage columns removed, consistent feature count across all three sets
- Added **sanity checks**: NaN counts, class balance per split, distribution drift check (all PASS)
- Added **preprocessing**: `LabelEncoder` fit exclusively on training set, applied to val/test (no leakage)

#### Visual EDA — 10 Charts (The Economist Style) saved to `eda_charts/`

| File | Description |
|---|---|
| `eda_01_class_imbalance.png` | Bar + 100% stacked horizontal bar. Debtor rate = **11.4%** |
| `eda_02_missingness.png` | **0 missing values** across all 37 columns |
| `eda_03_debtor_correlations.png` | Top 20 features split into negative (top) and positive (bottom) groups |
| `eda_04_feature_distributions.png` | Top 6 features — blue fill (Non-Debtor) + orange step outline (Debtor) |
| `eda_05_outliers.png` | Boxplots, top 8 features |
| `eda_06_correlation_heatmap.png` | Lower-triangle heatmap, top 15 features + Debtor |
| `eda_07_academic_performance.png` | Scatter: 1st vs 2nd sem approvals + grade violin |
| `eda_08_financial_indicators.png` | Debtor rate by Scholarship, Tuition status, International, Gender |
| `eda_09_age_enrollment.png` | Age distribution + debtor rate by age band |
| `eda_10_macroeconomic.png` | GDP, Unemployment, Inflation violins by debtor status |

#### Random Forest Model
- Trained `RandomForestClassifier` (n_estimators=100, class_weight=balanced)
- Added 5-fold cross-validation (`StratifiedKFold`) for stability assessment
- **Validation set:** Acc=0.890, F1=0.176, ROC-AUC=0.771
- **Hold-out test set (used once):** Acc=0.886, Precision=0.800, Recall=0.109, F1=0.192, ROC-AUC=0.784
- Top feature: `Tuition fees up to date` (importance=0.1204)

#### Autoencoder — Anomaly Detection (Stage 8)
- Framework: PyTorch; Architecture: `35 → 32 → 16 → 32 → 35` (MSE loss, Adam lr=0.001, early stopping patience=10)
- Trained on **Perfect Payers** (Debtor=0 AND Tuition fees up to date=1) → 3,639 students (82.3%)
- Alert threshold: 95th percentile of training reconstruction errors = 0.0296
- **Test results:** ROC-AUC=0.779, Precision@threshold=0.35, Recall=0.52, Alert rate=17.2%
- Charts saved: `ae_01_reconstruction_error.png`, `ae_02_roc_curve.png`, `ae_03_training_curve.png`

#### XGBoost + LightGBM Ensemble (soft-vote)
- Both models: n_estimators=300, lr=0.05
- **XGBoost:** Acc=0.872, Precision=0.485, Recall=0.427, F1=0.454, ROC-AUC=0.757
- **LightGBM:** Acc=0.863, Precision=0.451, Recall=0.455, F1=0.453, ROC-AUC=0.747
- **Untuned Ensemble:** Acc=0.860, Precision=0.445, Recall=0.518, F1=0.479, ROC-AUC=0.792
- Charts saved: `ens_01_roc_comparison.png`, `ens_02_metrics_bar.png`, `ens_03_feature_importance.png`

#### Fine-Tuned Ensemble (RandomizedSearchCV)
- n_iter=30, scoring=roc_auc, evaluated on validation set for both XGBoost and LightGBM
- **Tuned Ensemble:** Acc=0.860, Precision=0.445, Recall=0.518, F1=0.479, ROC-AUC=0.792
- **Key result:** 4.8× improvement in recall over base Random Forest (0.11 → 0.52)
- Tuned ensemble ROC-AUC (0.792) outperforms base Random Forest (0.784)

### Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.886 | 0.800 | 0.109 | 0.192 | 0.784 |
| Autoencoder ¹ | — | 0.349 | 0.520 | — | 0.779 |
| XGBoost | 0.872 | 0.485 | 0.427 | 0.454 | 0.757 |
| LightGBM | 0.863 | 0.451 | 0.455 | 0.453 | 0.747 |
| **Ensemble (tuned)** | **0.860** | **0.445** | **0.518** | **0.479** | **0.792** |

¹ *Autoencoder: Precision/Recall computed at 95th-percentile alert threshold. Accuracy and F1 not directly comparable.*

**Decision:** Tuned Ensemble is operationally superior — highest recall (52%) and highest ROC-AUC (0.792).

---

## 📝 How to Update This Log

When making changes to the project, append a new entry at the bottom of this file using the format:

```
## Session N — YYYY-MM-DD (~HH:MM UTC)

| Time (UTC) | Action |
|---|---|
| HH:MM:SS | Description of action taken |
```

---

## Session 6 — 2026-02-28 (~22:42 UTC)
**Source:** Current session — "Target Variable Correction"

| Time (UTC) | Action |
|---|
|---|
| 22:42:13 | Identified discrepancy: `Financial_Distress` (composite feature) was created in `Predictive_Indiv.ipynb` for EDA purposes only and was **never used** as the model target in `train_model.py`. All models were trained on `Debtor` directly. |
| 22:42:13 | **Decision:** Canonise `Debtor` as the sole model target. Rationale: using `Tuition fees up to date` inside a composite target while also including it as a predictor feature would introduce data leakage. |
| 22:42:13 | Updated `README.md`: replaced `Financial_Distress` target description with `Debtor`; added clarifying note about the EDA-only composite feature. |
| 22:42:13 | Updated `project_log.md` Session 1 entry: clarified `Financial_Distress` was created for EDA exploration only, not as the model target. |
| 22:42:13 | Updated `logs/actions.log`: added correction entry for Session 6. |

> **Impact on code:** Zero — `train_model.py` already uses `Debtor` throughout. No code changes required.

