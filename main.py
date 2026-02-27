import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("dataset/heart.csv")

# Basic cleaning
df = df.dropna()
df = df.drop_duplicates()

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    random_state=42
)

model.fit(X_train, y_train)

#Done creating and training model 

patient_1 = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 2
}])


patient_2 = pd.DataFrame([{
    "age": 46,
    "sex": 1,
    "cp": 2,
    "trestbps": 150,
    "chol": 231,
    "fbs": 0,
    "restecg": 1,
    "thalach": 147,
    "exang": 0,
    "oldpeak": 3.6,
    "slope": 1,
    "ca": 0,
    "thal": 2
}])

prediction = model.predict(patient_1)
probability = model.predict_proba(patient_1)


confidence = int(max(probability[0]) * 100)
print(confidence)
if prediction[0] == 1 and confidence > 70:
    print("⚠️ The patient is predicted to have heart disease.")
else:
    print("✅ The patient is predicted NOT to have heart disease.")