# 🎓 Predictive Individual Assignment
### Student Debtor Risk Prediction

---

## 📌 Project Overview

This project builds a machine learning model to predict whether a student is a **debtor** (i.e., has outstanding tuition debt), using demographic, academic, and socio-economic features from a higher education dataset.

### Target Variable
- **`Debtor`** — Binary classification target from the raw dataset (1 = student has outstanding debt, 0 = no debt)
- `Tuition fees up to date` is retained as a **predictor feature** (not part of the target)
- The original `Target` (dropout prediction) column has been dropped as a leakage risk

> **Note:** An earlier notebook exploration created a composite `Financial_Distress` feature (`Debtor OR fees not up to date`) for EDA purposes only. The final model uses `Debtor` exclusively as the target to avoid leakage from `Tuition fees up to date`.

---

## 📁 Project Structure

```
Predictive Indiv/
│
├── Predictive_Indiv.ipynb       # Main notebook (EDA + pipeline + modelling)
├── train_model.py               # Full model training script (all stages complete)
├── generate_report.py           # Report generation script
├── README.md                    # This file
├── project_log.md               # Full action log of all project changes
│
├── datasets/
│   └── student_data.csv         # Raw student dataset (CSV)
│
├── eda_charts/                  # 10 EDA charts (The Economist style)
│   ├── eda_01_class_imbalance.png
│   ├── eda_02_missingness.png
│   ├── eda_03_debtor_correlations.png
│   ├── eda_04_feature_distributions.png
│   ├── eda_05_outliers.png
│   ├── eda_06_correlation_heatmap.png
│   ├── eda_07_academic_performance.png
│   ├── eda_08_financial_indicators.png
│   ├── eda_09_age_enrollment.png
│   ├── eda_10_macroeconomic.png
│   ├── ae_01_reconstruction_error.png
│   ├── ae_02_roc_curve.png
│   ├── ae_03_training_curve.png
│   ├── ens_01_roc_comparison.png
│   ├── ens_02_metrics_bar.png
│   └── ens_03_feature_importance.png
│
├── logs/
│   └── actions.log              # Raw timestamped action log
│
└── PRedictive _ Individual Assignment.pdf   # Assignment brief
```

---

## 🔄 Pipeline Progress

| Stage | Description | Status |
|---|---|---|
| 1 | Problem Definition | ✅ Complete |
| 2 | Exploratory Data Analysis (EDA) | ✅ Complete |
| 3 | Data Preparation (60/20/20 split) | ✅ Complete |
| 4 | Model Training (Random Forest) | ✅ Complete |
| 5 | Autoencoder Anomaly Detection | ✅ Complete |
| 6 | Ensemble (XGBoost + LightGBM) + Fine-Tuning | ✅ Complete |
| 7 | Model Evaluation & Results | ✅ Complete |

---

## 🗂️ Dataset

- **Source:** UCI ML Repository (loaded via direct GitHub raw CSV URL)
- **Shape:** ~4,424 students × 37 features
- **Missing values:** 0 | **Duplicates:** 0
- **Debtor rate:** ~11.4% (class imbalance — handled via `class_weight='balanced'`)
- **Key preprocessing decisions:**
  - Dropped `Target` (dropout) column — **leakage risk**
  - 60% train / 20% validation / 20% test split via `StratifiedShuffleSplit`
  - `LabelEncoder` fit on train only, applied to val/test

---

## 🤖 Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.886 | 0.800 | 0.109 | 0.192 | 0.784 |
| Autoencoder ¹ | — | 0.349 | 0.520 | — | 0.779 |
| XGBoost | 0.872 | 0.485 | 0.427 | 0.454 | 0.757 |
| LightGBM | 0.863 | 0.451 | 0.455 | 0.453 | 0.747 |
| **Ensemble (tuned)** | **0.860** | **0.445** | **0.518** | **0.479** | **0.792** |

¹ *Autoencoder: Precision/Recall computed at 95th-percentile alert threshold.*

**Recommended model:** Tuned Ensemble (XGBoost + LightGBM soft-vote) — highest ROC-AUC (0.792) and 4.8× higher recall than base Random Forest (52% vs 11%).

---

## 📋 Key Decisions & Notes

- Switched from `ucimlrepo` library to direct CSV download for simplicity and reproducibility
- Financial distress (`Debtor`) used as target — more actionable than dropout prediction
- All EDA charts use **The Economist style** (red rule, white background, horizontal gridlines, blue/orange palette)
- Autoencoder trained only on **Perfect Payers** (Debtor=0 AND Tuition fees up to date=1) for anomaly detection
- Fine-tuning uses `RandomizedSearchCV` (n_iter=30) evaluated on the validation set — test set used exactly once

---

*Last updated: 2026-02-28 | See `project_log.md` for full change history.*
