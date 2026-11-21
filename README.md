# Credit Card Customer Attrition — Comparative Machine Learning & Ensemble Learning (6001CMD, Group 27)

A reproducible, leakage-safe machine learning workflow comparing **Logistic Regression**, **Random Forest**, **Gradient Boosting**, and a **Soft Voting Ensemble** on the public Kaggle *Credit Card Customers* (`BankChurners.csv`) dataset.

---

## Project Overview

This repository implements an end-to-end binary classification study (Parts A–G) distinguishing **Existing Customers (`0`)** from **Attrited Customers (`1`)**:

- **Part A — Problem Definition & Audit:** Verifies dataset dimensions, removes pretrained Naive Bayes leakage artefacts, and audits target class prevalence.
- **Part B — Exploratory Data Analysis (EDA):** Analyses descriptive statistics, distribution skewness, explicit (`NaN`) vs. implicit (`"Unknown"`) missingness, `1.5 × IQR` outliers, and Pearson correlations.
- **Part C — Leakage-Safe Preprocessing Pipeline:**
  - Stratified 80/20 train–test split (`8,101` train / `2,026` hold-out test, `random_state=42`).
  - Removes `CLIENTNUM`, `Naive_Bayes_Classifier_*`, and the algebraically redundant `Avg_Open_To_Buy`.
  - Engineers three deterministic row-wise features: `Avg_Spend_Per_Trans`, `Friction_Engagement_Score`, and `Revolving_Per_Product`.
  - Converts literal `"Unknown"` categories to missing values and performs all imputation, scaling, encoding, and mixed-type oversampling (**SMOTENC**) strictly **inside each 3-fold Stratified CV training fold** via `imblearn.pipeline.Pipeline`.
  - Evaluates preprocessing and resampling impact through a training-only CV ablation study.
- **Part D — Base Model Development:** Trains and benchmarks Logistic Regression, Random Forest, and Gradient Boosting under identical folds and preprocessing.
- **Part E — Hyperparameter Optimisation:** Tunes each pipeline on training-only 3-fold CV (`GridSearchCV` for Logistic Regression; `RandomizedSearchCV` for Random Forest and Gradient Boosting) optimising positive-class **F1-score**.
- **Part F — Ensemble Learning:** Selects the top two tuned models strictly by training CV F1 (**Gradient Boosting** and **Random Forest**), excludes Logistic Regression, and combines them using equal-weight **Soft Voting**.
- **Part G — Independent Hold-Out Evaluation:** Evaluates all four frozen models once on the untouched 2,026-record test set, including ROC curves, confusion matrices, fit/inference runtime, and a 500-resample paired bootstrap 95% confidence interval.

---

## Dataset

- **Source:** [Credit Card Customers (`BankChurners.csv`) on Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) (Goyal, n.d.)
- **Size:** `10,127` customer records × `23` raw columns (`21` audited attributes after dropping the two pretrained Naive Bayes columns)
- **Class Imbalance:** `8,500` Existing Customers (`83.93%`) vs. `1,627` Attrited Customers (`16.07%`) — approximately `5.2:1`

> **Setup Note:** Place `BankChurners.csv` in the repository root directory alongside `6001CMD_CW2_Source.py` before running.

---

## Final Independent Hold-Out Results (`n = 2,026`)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC (AP) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** *(Recommended)* | **0.9650** | **0.8896** | **0.8923** | **0.8909** | 0.9886 | **0.9554** | 13.05 |
| **Soft Voting (GB + RF)** | 0.9585 | 0.8731 | 0.8677 | 0.8704 | **0.9889** | 0.9549 | 18.08 |
| **Random Forest** | 0.9506 | 0.8483 | 0.8431 | 0.8457 | 0.9841 | 0.9354 | 4.89 |
| **Logistic Regression** | 0.8687 | 0.5648 | 0.7908 | 0.6590 | 0.9243 | 0.7405 | 0.92 |

### Key Findings
- **Gradient Boosting** achieved the strongest overall hold-out performance (`F1 = 0.8909`, `Recall = 0.8923`, `Precision = 0.8896`), detecting `290 / 325` attrited customers with `36` false positives and `35` false negatives.
- **Soft Voting** underperformed standalone Gradient Boosting by `-0.0206` in hold-out F1 (500-resample paired bootstrap 95% CI: `[-0.0377, -0.0054]`), while requiring longer training and inference times.

---

## Repository Structure

```text
.
├── BankChurners.csv                       # Raw Kaggle dataset (place in root)
├── 6001CMD_CW2_Source.py                  # Standalone reproducible Python script
├── 6001CMD_CW2_Notebook.ipynb             # Executed Jupyter / Colab notebook
├── README.md                              # Project documentation
└── requirements.txt                       # Python dependencies
```

### Generated Outputs
Running the script or notebook generates the following artefacts in the working directory:
- `BankChurners_Processed_FIXED.csv` — Numeric, non-synthetic processed dataset (`10,127 × 38`) fitted on the training split only. **Do not feed this file back into cross-validation.**
- `PartG_Final_Model_Comparison.csv` — Complete hold-out evaluation table.
- `PartD_Baseline_Feature_Importance.png` — Baseline generalisation gap and tree feature importances.
- `PartE_CV_Tuning.png` — Before-vs-after hyperparameter tuning and generalisation gaps.
- `PartF_Ensemble_CV_Comparison.png` — Pre-test CV F1 comparison across tuned models and ensemble.
- `PartG_Final_ROC_and_Confusion_Matrices.png` — Hold-out ROC curves and confusion matrices.
- `PartG_F1_And_Runtime_Comparison.png` — Hold-out F1 and full-training fit duration comparison.

---

## Installation & Usage

### 1. Clone the Repository & Install Dependencies
Tested with Python `3.10+` (developed on Python `3.13.5`).

```bash
git clone <your-repository-url>
cd <your-repository-folder>

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run via Python Script (Recommended for Local Execution)
Ensure `BankChurners.csv` is in the project root, then run:

```bash
python 6001CMD_CW2_Source.py
```

### 3. Run via Jupyter Notebook / Google Colab
1. Open the `.ipynb` file in Jupyter Lab/Notebook or upload it to Google Colab.
2. Upload or place `BankChurners.csv` in the expected path (in Colab, update `DATA_PATH` if placed in `/content/BankChurners.csv` rather than `/BankChurners.csv`).
3. Select **Runtime → Restart and run all**.

---

## Scope, Limitations & Governance

- **Retrospective Study:** The public dataset lacks temporal cutoff documentation; predictors such as `Total_Trans_Ct` may overlap with the period in which attrition occurred. Temporal validation is required before operational deployment.
- **No Verified Cost Matrix:** The `0.85` recall reference point is illustrative only. Decision thresholds and probability calibration should be validated against real bank contact costs and retention value.
- **Fairness & Privacy:** Subgroup fairness (e.g., across age or gender) was not audited in this coursework and must be evaluated prior to real-world use.