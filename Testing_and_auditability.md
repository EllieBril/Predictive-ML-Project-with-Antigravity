# Model Testing, Reproducibility, and Auditability Strategy

In data science, a machine learning model is only as reliable as the pipeline that produces it. Unlike traditional software engineering where code either runs or crashes, machine learning systems are prone to **silent failures**: the code executes perfectly, but the data has drifted, a feature has leaked the target variable, or the model logic has fundamentally degraded. 

This document outlines the strict testing, reproducibility, and auditability framework implemented for the Student Debtor Risk prediction model.

---

## 1. The Importance of Data Science Testing

Traditional software unit tests ensure that `2 + 2 = 4`. Data science testing must also ensure that the data being fed into the equation is valid, and that the model's predictive behavior aligns with domain reality.

If a pipeline lacks rigorous MLOps testing:
- **Schema Drift:** A suddenly missing column or a change in data type (e.g., `Age` arriving as a string instead of an integer) will crash production silently or cause unpredictable model behavior.
- **Data Leakage:** A feature that implicitly contains the future target (e.g., `Target` status) might accidentally be included in the training set, resulting in artificially perfect evaluation metrics that instantly collapse in the real world.
- **Model Degradation:** A newly retrained model might have a severe blind spot for a specific demographic, or its overall AUC might drop below an acceptable threshold due to shifting student profiles over time.

To combat this, we have implemented four distinct layers of programmatic testing in `train_model.py`.

---

## 2. Testing Taxonomy Implemented

The pipeline automatically executes the following test suite (Section 11) before allowing final model evaluation to proceed. If any of these assertions fail, the pipeline *intentionally crashes* to prevent a degraded model from being deployed.

### A. Data Tests (Schema & Bounds)
Data tests act as a strict gatekeeper for the incoming dataset.
- **Non-null Assertions:** Verifies that critical core columns (`Tuition fees up to date`, `Age at enrollment`, `Debtor`) contain zero missing values.
- **Distributional Bounds:** Ensures numerical logic (e.g., `Age at enrollment` > 0).
- **Schema Enforcement:** Confirms the target `Debtor` column is strictly binary (0 or 1).

### B. Unit Tests (Correctness)
Unit tests ensure custom mathematical helper functions return expected results.
- **Metric Verification:** We pass a mocked, deterministic dataset (with known True Positives, False Positives, etc.) into our custom `_subgroup_metrics` function to explicitly assert that the calculated Recall and Precision match our manual math.

### C. Integration Tests (Leakage Prevention)
Integration tests confirm that the boundaries between different stages of the pipeline remain intact.
- **Row-level Disjointness:** Mathematically asserts that the intersection of indices between the Training, Validation, and Test sets is exactly zero (no rows appear in more than one set).
- **Column-level Hygiene:** Hard-checks that the target variable `Debtor` strictly does not exist in the `X_train`, `X_val`, or `X_test` feature matrices prior to model fitting.

### D. Model Tests (Behavioral Guardrails)
Model tests evaluate the actual intelligence of the trained artifact.
- **Performance Floor:** The model's ROC-AUC on the un-seen test set must exceed a hard-coded safety threshold (e.g., 0.77). If it drops below this, the primary predictive signals have corrupted.
- **Directional Invariance (Monotonicity):** We extract an "average" student profile and ask the model to predict their risk twice: once with their tuition *paid*, and once with it *unpaid*. The test asserts that the model MUST predict higher debtor risk for the unpaid profile. This guarantees the model's logic aligns with basic financial reality.

---

## 3. Reproducibility

Reproducibility ensures that another data scientist can run the same code on the same data and get the exact same metrics and model weights.

**Measures taken:**
1. **Deterministic Randomness:** All stochastic processes (train/test splitting, Random Forest tree building, ensemble bootstrapping) use a fixed `random_state=42`.
2. **Explicit Validation Set:** We transitioned from a simple Train/Test split to a rigorous 60/20/20 Train/Validation/Test split using `StratifiedShuffleSplit`. The test set is locked away and touched exactly *once* at the very end of the script.
3. **Automated Hyperparameter Tuning:** Using `RandomizedSearchCV` with a `PredefinedSplit`, the tuning phase is entirely algorithmic and documented, removing human bias from the parameter selection process.
4. **Environment:** Executing the pipeline relies solely on standard, versionable libraries (`scikit-learn`, `pandas`, `xgboost`, `lightgbm`).

---

## 4. Auditability

An auditable model is one whose decisions, biases, and failure modes are highly transparent to stakeholders.

**Measures taken:**
1. **Demographic Bias Audit (Section 9):** Calculates Disparate Impact and Equal Opportunity Difference across Gender, Age, International status, and Scholarship status to explicitly document algorithmic fairness.
2. **Robust Error Analysis (Section 12):** Beyond aggregate numbers, the pipeline generates:
    - **Confusion Matrices**: To show raw error counts and the exact cost of False Alarms vs Missed Debtors.
    - **Calibration Curves**: To prove whether a model's "80% risk" score actually corresponds to an 80% real-world likelihood.
    - **Failure Mode Profiling**: A unique process that isolates the model's False Negatives (the debtors it missed) and compares their demographic/academic profile to the debtors it caught, revealing the model's specific blind spots (e.g., missing students who have high early grades but fail later).
