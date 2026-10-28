# CreditIQ — Credit Scoring Model

A machine learning project that predicts the likelihood of serious credit delinquency using Logistic Regression and Random Forest. It includes data preprocessing, model evaluation, comparison, and an interactive Streamlit web application.

## Overview

CreditIQ analyzes financial and credit-history indicators to predict whether a person may experience serious delinquency within the next two years, based on the target definition in the dataset.

The project compares two classification algorithms and provides an interactive dashboard where users can enter financial information and view the model's prediction.  

## Features

- Data loading and exploratory data analysis (EDA)
- Missing-value handling using median imputation
- Feature scaling for Logistic Regression
- Class imbalance handling using `class_weight="balanced"`
- Logistic Regression and Random Forest model training
- Model comparison using classification metrics
- Model persistence using Joblib
- Interactive credit-risk prediction dashboard built with Streamlit

## Technologies Used

- Python
- Pandas
- NumPy 
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset

This project uses the **Give Me Some Credit** dataset from Kaggle.

The dataset contains financial and credit-history features, including:

- Age
- Monthly income
- Debt ratio
- Revolving utilization of unsecured credit lines
- Number of open credit lines and loans
- Previous late-payment records
- Number of dependents

The target variable is `SeriousDlqin2yrs`:

- `0` — No recorded serious delinquency within two years
- `1` — Serious delinquency recorded within two years

The training dataset used in this project contains 104,805 rows and 12 columns, including the target and record ID.

## Project Structure

```text
CodeAlpha_CreditScoring/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
│
├── notebooks/
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Machine Learning Models

### 1. Logistic Regression

A classification algorithm used as a baseline model. Median imputation, feature scaling, and class weighting are applied through a Scikit-learn pipeline.

### 2. Random Forest

An ensemble learning algorithm that combines multiple decision trees to produce predictions. The implementation uses median imputation and class weighting to address missing values and class imbalance.

## Model Evaluation

The models were evaluated using a stratified 80/20 train-test split.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 75.90% | 85.01% |
| Precision — Class 1 | 16.34% | 25.87% |
| Recall — Class 1 | 63.96% | 67.55% |
| F1-score — Class 1 | 26.04% | 37.41% |
| ROC-AUC | 0.7832 | 0.8595 |

### Results

Random Forest performed better than Logistic Regression across the reported metrics on the held-out test set.

However, the precision for class 1 is relatively low, meaning many positive predictions are false positives. Accuracy alone is not sufficient to assess a credit-risk model, especially when the target classes are imbalanced.

These results are specific to the current dataset, split, and model settings; they do not guarantee performance on new data.

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_CreditScoring.git
cd CodeAlpha_CreditScoring
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the Give Me Some Credit dataset from Kaggle and place the required CSV files in the `data/` directory.

Ensure that `data/train.csv` is available before training.

### 4. Train and evaluate the models

```bash
python src/train.py
```

This script trains both models, evaluates their performance, compares the results, and saves the trained pipelines in the `models/` directory.

### 5. Test a prediction

```bash
python src/predict.py
```

This script loads the saved Random Forest model and predicts the outcome for a sample financial profile.

### 6. Launch the web application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the CreditIQ dashboard.

## Important Limitations

- The dataset has an imbalanced target distribution.
- Predictions may be incorrect and can produce false positives or false negatives.
- Model-estimated probabilities should not be assumed to be calibrated.
- The model has not been established as suitable for real-world lending decisions.
- The project is intended for educational purposes and machine learning practice.

**This application must not be used to approve or reject real loan applications or make actual financial decisions.**

## Future Improvements

- Hyperparameter tuning and cross-validation
- Probability calibration and decision-threshold analysis
- Feature-importance visualization and model explainability
- More comprehensive testing and validation
- Improved handling of outliers and unusual input values
- Deployment to a hosting platform

## Acknowledgements

- Kaggle — Give Me Some Credit dataset
- Scikit-learn — Machine learning tools
- Streamlit — Interactive web application framework

## Author

Developed as part of a Machine Learning internship project with CodeAlpha.

---

*CreditIQ — Credit Risk Analytics powered by Machine Learning.*
