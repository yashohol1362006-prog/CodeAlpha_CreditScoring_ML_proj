import joblib
import pandas as pd
from pathlib import Path

# Find the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load the saved Random Forest model
model_path = PROJECT_DIR / "models" / "random_forest.joblib"
model = joblib.load(model_path)

# Example input: replace with actual feature values
sample = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": 0.25,
    "age": 35,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.30,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1.0
}])

# Predict the class and probability
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0, 1]

print("Predicted class:", prediction)
print("Estimated model probability:", round(probability, 4))