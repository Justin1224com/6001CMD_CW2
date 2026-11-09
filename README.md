# Telecommunications Customer Churn Prediction Pipeline
**Supervised Binary Classification & Heterogeneous Ensemble Learning**

## 1. Project Overview
This repository contains the end-to-end Machine Learning Consultancy pipeline developed to predict subscriber attrition using the **IBM Telco Customer Churn Dataset** (N = 7,043 records, 21 raw attributes)[cite: 4]. With a baseline churn rate of **26.54%** (1,869 churned vs. 5,174 retained subscribers) and an average monthly charge of **$64.76**, customer attrition exposes the business to **$121,039.60/month** (~$1.45M/year) in lost recurring revenue[cite: 4].

To support proactive, targeted retention interventions, this pipeline implements a leak-free preprocessing workflow, evaluates three mathematically diverse base classifiers, performs 5-Fold Stratified Hyperparameter Optimisation, and constructs a calibrated **Weighted Soft Voting Ensemble (Gradient Boosting + Logistic Regression)**[cite: 1, 2, 3, 4].



## 2. Repository Structure
* `WA_Fn-UseC_-Telco-Customer-Churn.csv` — Raw IBM Telco Customer Churn dataset (7,043 rows x 21 columns)[cite: 4]
* `Telco_Customer_Churn_Processed.csv` — Cleaned, engineered & encoded dataset (7,043 rows x 34 columns)[cite: 4]
* `Telco_Customer_Churn_Notebook.ipynb` — Fully executed Jupyter Notebook (Parts B to G)[cite: 4]
* `Telco_Customer_Churn_Pipeline.py` — Standalone Python source script (.py)
* `requirements.txt` — Python package dependencies
* `README.md` — Project documentation and execution guide
* `figures/` — High-resolution (300 DPI) evaluation plots (`PartB_Fig1` through `PartG_Fig6`)[cite: 1, 3, 4]



## 3. Methodology & Pipeline Architecture
1. **Part A & B — Problem Formulation & Exploratory Data Analysis (EDA):**
   * Identified 11 structural whitespace missing values (`" "`) in `TotalCharges` where `tenure == 0` (MNAR)[cite: 4].
   * Audited multicollinearity (`tenure` vs. `TotalCharges`, r = 0.826) and high-risk segments (Month-to-month contracts: 42.71% churn; Electronic check: 45.29% churn; Fiber optic: 41.89% churn)[cite: 4].
2. **Part C — Leak-Free Data Preparation:**
   * Imputed structural `TotalCharges` nulls with `$0.00` and dropped non-informative `customerID`[cite: 4].
   * Engineered 3 domain features: `Total_Addons`, `Charge_Difference`, and `Avg_Monthly_Spend`[cite: 4].
   * Applied binary mapping and drop-first One-Hot Encoding (33 predictors + 1 target = 34 columns)[cite: 4].
   * Executed an 80:20 Stratified Train-Test Split (`random_state=42`: N_train = 5,634, N_test = 1,409)[cite: 4].
   * Fitted `StandardScaler` (on the 6 continuous features) and `SMOTE` (N_train_smote = 8,278, 50:50 class balance) **strictly on the training partition** to prevent data leakage[cite: 4].
3. **Part D & E — Base Model Development & 5-Fold Stratified Hyperparameter Tuning:**
   * Developed and tuned **Logistic Regression** (`GridSearchCV`), **Random Forest** (`RandomizedSearchCV`), and **Gradient Boosting** (`RandomizedSearchCV`)[cite: 1, 4].
   * Tree depth and leaf regularisation reduced Random Forest overfitting (`Train F1 - Test F1`) by **41.9%** (+0.4105 to +0.2384), boosting its Test Recall by **+12.30%** (0.6203 to 0.7433) and Test ROC-AUC to **0.8406**[cite: 1, 4].
4. **Part F & G — Heterogeneous Ensemble & Final Comparative Evaluation:**
   * Constructed a **Weighted Soft Voting Classifier (`Gradient Boosting + Logistic Regression`, w=[2, 1])** combining non-linear tree interactions with calibrated linear log-odds[cite: 2, 4].


## 4. Summary of Final Empirical Results (Unseen Test Set, N = 1,409)

| Model Architecture | Train F1 (SMOTE) | Test Acc | Test Prec | Test Recall | Test F1 | Test ROC-AUC | Overfit Gap | TN / FP / FN / TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base 1: Logistic Regression (Tuned)** | 0.8238 | 0.7658 | **0.5449** | 0.7139 | 0.6181 | 0.8362 | **+0.2057** | 812 / 223 / 107 / 267 |
| **Base 2: Random Forest (Tuned)** | 0.8659 | 0.7658 | 0.5430 | 0.7433 | **0.6275** | **0.8406** | +0.2384 | 801 / 234 / 96 / 278 |
| **Base 3: Gradient Boosting (Tuned)** | 0.8331 | 0.7622 | 0.5376 | **0.7460** | 0.6249 | 0.8382 | +0.2082 | 795 / 240 / **95** / **279** |
| **Ensemble: Soft Voting (GB + LR)** | 0.8382 | **0.7665** | 0.5446 | 0.7353 | 0.6257 | **0.8406** | +0.2125 | 805 / 230 / 99 / 275 |

* **Deployment Recommendation:** **Ensemble: Soft Voting (`GB + LR`, w=[2,1])** is selected for production deployment, achieving the highest overall Test Accuracy (**0.7665**), joint-highest Test ROC-AUC (**0.8406** >= 0.84 benchmark), highest 5-Fold CV F1 (**0.8303**), low inference latency (**10.29 ms**), and **$17,809.00/month** in protected test-set recurring revenue[cite: 2, 3, 4].


## 5. How to Run the Code
1. Clone this repository and navigate to the project folder.
2. Install required dependencies: `pip install -r requirements.txt`
3. Open and run `Telco_Customer_Churn_Notebook.ipynb` in Google Colab or Jupyter Lab, or run the standalone Python script via `python Telco_Customer_Churn_Pipeline.py`.
