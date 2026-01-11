import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train():
    # 1. LOAD
    try:
        df = pd.read_csv("data/student_data.csv")
    except FileNotFoundError:
        print("Error: File data/student_data.csv not found. Run generate_data.py first!")
        return

    # 2. CLEANING - even if data is clean, this stage should always be in production
    # (In our case data is already clean, so I'll add it later)
    
    # 3. ENCODING
    df = pd.get_dummies(df, columns=['Subject'], drop_first=True)
    X = df.drop(columns=['Grade', 'Student_ID']) 
    y = df['Grade']
    
    # 4. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. TRAIN
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. EVALUATE
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Model Performance -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # 7. SAVE
    os.makedirs("models", exist_ok=True)
    artifact = {
        "model": model,
        "features": X.columns.tolist()
    }
    joblib.dump(artifact, "models/grade_model.pkl")
    print("Model trained and saved to models/grade_model.pkl")

if __name__ == "__main__":
    train()