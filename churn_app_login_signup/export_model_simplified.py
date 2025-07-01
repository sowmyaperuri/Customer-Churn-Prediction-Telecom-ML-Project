import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Cleaned_Telco_Customer_Churn.csv")

# Drop unneeded column
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Keep only required columns
features_to_keep = [
    'gender', 'SeniorCitizen', 'Partner', 'tenure',
    'MonthlyCharges', 'Contract', 'InternetService', 'PaperlessBilling', 'Churn'
]
df = df[features_to_keep]

# Convert binary Yes/No columns to 1/0
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})

# One-hot encode multi-category columns
df = pd.get_dummies(df, columns=['gender', 'Contract', 'InternetService'], drop_first=True)

# Split features and labels
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models and select the best
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    score = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"{name} Accuracy: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = model

# Save best model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ model.pkl and scaler.pkl created using simplified feature set.")
