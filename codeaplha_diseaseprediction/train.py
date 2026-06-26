import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression

# Define the top 5 most important columns for the heart model
HEART_FEATURES = ['age', 'cp', 'thalach', 'oldpeak', 'ca']

# ==========================================
# 1. TRAIN DIABETES MODEL
# ==========================================
print("Training Diabetes Model...")
try:
    diabetes_data = pd.read_csv("diabetes.csv")
    X_diabetes = diabetes_data.drop(columns=['Outcome']) 
    y_diabetes = diabetes_data['Outcome']

    X_train_diab, X_test_diab, y_train_diab, y_test_diab = train_test_split(
        X_diabetes, y_diabetes, test_size=0.2, random_state=42
    )

    diabetes_model = RandomForestClassifier(random_state=42)
    diabetes_model.fit(X_train_diab, y_train_diab)
    joblib.dump(diabetes_model, "model/diabetes.pkl")
    print("✅ Diabetes model saved to 'model/diabetes.pkl'")
except Exception as e:
    print(f"❌ Error training Diabetes model: {e}")


# ==========================================
# 2. TRAIN HEART DISEASE MODEL (REDUCED FEATURES)
# ==========================================
print("\nTraining Heart Disease Model...")
try:
    heart_data = pd.read_csv("heart.csv")
    
    # Filter to only keep the 5 main important features
    X_heart = heart_data[HEART_FEATURES] 
    y_heart = heart_data['target']

    X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(
        X_heart, y_heart, test_size=0.2, random_state=42
    )

    heart_model = LogisticRegression(max_iter=1000, random_state=42)
    heart_model.fit(X_train_heart, y_train_heart)
    joblib.dump(heart_model, "model/heart.pkl")
    print("✅ Heart model saved to 'model/heart.pkl' using the top 5 key features!")
except Exception as e:
    print(f"❌ Error training Heart Disease model: {e}")


# ==========================================
# 3. TRAIN BREAST CANCER MODEL
# ==========================================
print("\nTraining Breast Cancer Model...")
try:
    cancer_data = pd.read_csv("breast.csv", na_values='?') 
    cancer_data = cancer_data.dropna()

    for id_col in ['id', 'id_number', 'Sample_code_number']:
        if id_col in cancer_data.columns:
            cancer_data = cancer_data.drop(columns=[id_col])

    X_cancer = cancer_data.iloc[:, :-1]  
    y_cancer = cancer_data.iloc[:, -1]   

    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
        X_cancer, y_cancer, test_size=0.2, random_state=42
    )

    X_train_cancer = X_train_cancer.astype(float)
    X_test_cancer = X_test_cancer.astype(float)

    cancer_model = RandomForestClassifier(random_state=42)
    cancer_model.fit(X_train_cancer, y_train_cancer)

    joblib.dump(cancer_model, "model/cancer.pkl")
    print("✅ Breast Cancer model saved to 'model/cancer.pkl'")
except Exception as e:
    print(f"❌ Error training Breast Cancer model: {e}")

print("\n🎉 Process complete! Check your 'model/' directory.")