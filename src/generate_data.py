import pandas as pd
import numpy as np
import os

def generate_smart_data(n_rows=20000):
    np.random.seed(42)
    
    study_hours = np.random.normal(15, 5, n_rows).round(1)
    study_hours = np.maximum(study_hours, 0)
    
    social_hours = np.random.exponential(10, n_rows).round(1)
    
    subjects = ['Math AA HL', 'Physics HL', 'Chem HL', 'Eng A Lit SL', 'French B SL', 'Econ SL']
    subj_choices = np.random.choice(subjects, n_rows)
    
    base_intelligence = np.random.normal(0, 1, n_rows) 
    
    difficulty_map = {
        'Math AA HL': -1.5,
        'Physics HL': -1.2,
        'Chem HL': -0.5,
        'Eng A Lit SL': 0.0,
        'Econ SL': 0.5,
        'French B SL': 0.8
    }
    
    subject_effect = np.array([difficulty_map[s] for s in subj_choices])
    
    raw_grade = 4.0 + (0.15 * study_hours) - (0.05 * social_hours) + subject_effect + (0.5 * base_intelligence) + np.random.normal(0, 0.5, n_rows)
    
    grades = np.clip(np.round(raw_grade), 1, 7)
    
    df = pd.DataFrame({
        'Student_ID': range(n_rows),
        'Subject': subj_choices,
        'Study_Hours': study_hours,
        'Social_Hours': social_hours,
        'Grade': grades
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    print("Generating data...")
    df = generate_smart_data(20000)
    
    save_path = "ib_grades_predictor/data/student_data.csv"
    df.to_csv(save_path, index=False)
    
    print(f"Data successfully saved to {save_path}")
    print(df.head())