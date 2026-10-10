import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(accuracy_score, classification_report,
                            confusion_matrix,
                            roc_auc_score)


# Find the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load the training data
data_path = PROJECT_DIR / "data" / "train.csv"
df = pd.read_csv(data_path)

# Display basic information
print(df.head())
print("Dataset shape:", df.shape)
print(df.info())
print(df.isnull().sum())

# checking target class distribution 
print('\nTarget ditribution')
print(df['SeriousDlqin2yrs'].value_counts())

# Visualization of SeriousDlqin2yrs

# counting target classes
target_counts = df['SeriousDlqin2yrs'].value_counts().sort_index()
print(target_counts)

target_counts.plot(kind= 'bar')
plt.title('Credit Delinquency Class Distribution')
plt.xlabel('Target (0= No recoreded delinquesy, 1= serous delinquency )')
plt.ylabel('Number of Records')
plt.xticks(rotation= 0)
plt.tight_layout()


# plt.show()

X = df.drop(columns=['SeriousDlqin2yrs', 'Id'])
y = df['SeriousDlqin2yrs']

print('Features shape :', X.shape)
print('Target shape :', y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('Training Features:', X_train.shape)
print('Testing Features:', X_test.shape)

print('training target distribution:')
print(y_train.value_counts(normalize=True).round(3))

print("Testing target distribution:")
print(y_test.value_counts(normalize=True).round(3))

# ML Model...

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Training the Models
print('\nTraining Logistic Regression...')
model.fit(X_train, y_train)

# Making prediction
y_pred= model.predict(X_test)
y_prob= model.predict_proba(X_test)[:, 1] #ex.>> [0.90, 0.10] >>means probab of class 0 is 90% and class 1 is 10%


# evaluate the model
print('\nAccuracy:')
print(accuracy_score(y_test, y_pred))

print('\nClassification Report:')
print(classification_report(y_test, y_pred, zero_division=0))

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nROC-AUC:')
print(roc_auc_score(y_test, y_prob))

# Creating Random Forest Pipeline
from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# 2. Train Random Forest
print("\nTraining Random Forest...")
rf_model.fit(X_train, y_train)

# 3. Make predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# 4. Evaluate Random Forest
print("\n===== RANDOM FOREST RESULTS =====")

print("Accuracy:", accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, rf_pred, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nROC-AUC:", roc_auc_score(y_test, rf_prob))

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_pred)
    ],
    "Precision (Class 1)": [
        precision_score(y_test, y_pred, zero_division=0),
        precision_score(y_test, rf_pred, zero_division=0)
    ],
    "Recall (Class 1)": [
        recall_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, rf_pred, zero_division=0)
    ],
    "F1-score (Class 1)": [
        f1_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, rf_pred, zero_division=0)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, rf_prob)
    ]
})

print("\n===== MODEL COMPARISON =====")
print(comparison.round(4).to_string(index=False))



import joblib

# Create the models directory if it doesn't exist
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Save both trained models
joblib.dump(
    model,
    MODEL_DIR / "logistic_regression.joblib"
)

joblib.dump(
    rf_model,
    MODEL_DIR / "random_forest.joblib"
)

print("\nModels saved successfully!")


import joblib

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "logistic_regression.joblib",
    compress=3
)

joblib.dump(
    rf_model,
    MODEL_DIR / "random_forest.joblib",
    compress=3
)

print("Compressed models saved successfully!")