import joblib
import pandas as pd
import numpy as np

def predict():
    # 1. DATA COLLECTION (User Input)
    try:
        study = float(input("Hours of study per week: "))
        social = float(input("Hours on social media: "))
        print("Select your subject:")
        subjects_list = ['Math AA HL', 'Physics HL', 'Chem HL', 'Eng A Lit SL', 'French B SL', 'Econ SL']
        for i, sub in enumerate(subjects_list, 1):
            print(f"[{i}] {sub}")
        
        while True:
            try:
                choice = int(input("Enter the number of your subject: "))
                if 1 <= choice <= len(subjects_list):
                    subject = subjects_list[choice - 1]
                    print(f"Selected: {subject}")
                    break
                else:
                    print(f"Please enter a number between 1 and {len(subjects_list)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    except ValueError:
        print("Error: Invalid input data!")
        return

    # 2. ENGINEERING MAGIC (Feature Alignment)
    # Create empty DataFrame with RIGHT columns, filled with zeros
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill numeric fields
    input_df['Study_Hours'] = study
    input_df['Social_Hours'] = social
    
    # Fill One-Hot field
    subject_col = f"Subject_{subject}"
    
    if subject_col in input_df.columns:
        input_df[subject_col] = 1
    # else:
    #     print(f"Note: Subject '{subject}' is considered basic (or entered incorrectly).")

    # 3. PREDICTION
    prediction = model.predict(input_df)[0]
    
    # 4. OUTPUT (limit from 1 to 7)
    final_grade = np.clip(prediction, 1, 7)
    print(f"\n🔮 Your predicted grade: {final_grade:.2f}")

if __name__ == "__main__":
    # LOAD "Brains"
    try:
        artifact = joblib.load("models/grade_model.pkl")
        model = artifact["model"]
        feature_names = artifact["features"] # ['Study_Hours', 'Social_Hours', 'Subject_Math...', ...]
    except FileNotFoundError:
        print("Error: Model not found.")


    print("--- IB Grade Predictor ---")

    while True:
        predict()
        
        again = input("\nDo you want to predict another grade? (yes/no): ").strip().lower()
        if again not in ["y", "yes", "ye", "yeah"]:
            break