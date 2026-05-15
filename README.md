# Liver Cirrhosis Stage Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## Project Overview
This project applies advanced machine learning techniques to predict the histologic stage of liver disease (Primary Biliary Cholangitis) using a patient's lifestyle and medical history. The goal is to accurately classify patients into Stages 1 through 4, with Stage 4 indicating clinical Cirrhosis.

This repository features a highly rigorous, production-ready machine learning pipeline that prioritizes **data leakage prevention**, **class imbalance handling**, and **nested cross-validation**.

## The Dataset
The data used in this project is the **Cirrhosis Prediction Dataset** (available on Kaggle). It contains medical records including continuous variables (Age, Bilirubin, Copper, Platelets) and categorical variables (Drug type, Presence of Ascites, Hepatomegaly).

## Methodology & Machine Learning Pipeline
To ensure the models represent real-world clinical accuracy, the following pipeline was implemented:

1. **Exploratory Data Analysis (EDA):** Analyzed feature distributions and utilized logarithmic box plots to identify severe outliers in clinical measurements like Bilirubin.
2. **Data Leakage Prevention:** All data scaling (`MinMaxScaler`), outlier capping (IQR Method), and missing value imputation (Median/Mode) were executed **strictly inside the cross-validation loop**. The models never "peeked" at future test data.
3. **Class Imbalance Handling:** Addressed the heavy skew toward late-stage patients by implementing `class_weight='balanced'` and utilizing `StratifiedKFold`.
4. **Nested Cross-Validation:** Combined 5-Fold Stratified Cross-Validation with `RandomizedSearchCV` to dynamically tune hyperparameter grids for every fold.

## Models Evaluated
* **Random Forest Classifier** (Tuned for `n_estimators`, `max_depth`, `min_samples_split`)
* **XGBoost Classifier** (Tuned for `learning_rate`, `max_depth`, `subsample`)

## Key Results & Insights

### Feature Importance
By analyzing the `.feature_importances_` of the tuned models, we identified the top clinical indicators driving disease progression. Both models heavily prioritized specific liver function tests.

*(Upload your `feature_importance_final.png` to your repo, and it will appear here!)*
![Top Features](feature_importance_final.png)

### Model Performance
*(Upload your `model_comparison.png` to your repo, and it will appear here!)*
![Model Performance](model_comparison.png)

* **Clinical Insight:** The models demonstrated that baseline clinical measurements (specifically Copper, Prothrombin, and Bilirubin) are robust predictors of late-stage cirrhosis.

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/Liver-Disease-Prediction.git](https://github.com/YourUsername/Liver-Disease-Prediction.git)
