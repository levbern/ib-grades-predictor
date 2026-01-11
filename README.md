# IB Grades Predictor

A machine learning-powered CLI tool designed to predict International Baccalaureate (IB) grades based on student habits and subject choices. This project demonstrates a complete data science workflow, from synthetic data generation and model training to an interactive prediction interface.

## 🚀 Features

- **Synthetic Data Engine:** Generates realistic student performance data based on weighted variables (study hours, social media usage, and subject difficulty).
- **Machine Learning Pipeline:** Utilizes a Linear Regression model to find correlations and predict outcomes.
- **Interactive CLI:** A user-friendly command-line interface with menu-driven selections for error-free input.
- **Model Persistence:** Saves and loads trained models using `joblib` for immediate inference without retraining.

## 🛠️ Technologies Used

- **Python 3**
- **Scikit-learn:** Model training and evaluation.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical computations.
- **Joblib:** Model serialization.

## 📋 Project Structure

```text
├── data/               # Raw and generated datasets (.csv)
├── models/             # Serialized model artifacts (.pkl)
├── src/                # Core source code
│   ├── generate_data.py # Data synthesis script
│   ├── train.py        # Model training and evaluation
│   └── predict.py      # Interactive prediction tool
├── README.md           # Project documentation
```

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/levbern/ib-grades-predictor.git
   cd ib_grades_predictor
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

Follow these steps in order to set up and run the predictor:

### 1. Generate Training Data
Create a synthetic dataset of 20,000 students to train the model.
```bash
python3 src/generate_data.py
```

### 2. Train the Model
Train the Linear Regression model and evaluate its performance (MAE & RMSE).
```bash
python3 src/train.py
```

### 3. Run Predictions
Launch the interactive tool to predict your own IB grade.
```bash
python3 src/predict.py
```

## 🧠 How it Works

The model calculates a predicted grade (1.0 to 7.0) based on:
- **Study Hours:** Positive correlation with grades.
- **Social Media Hours:** Negative correlation with grades.
- **Subject Difficulty:** Each subject has a specific difficulty weight (e.g., Math AA HL vs. French B SL).
- **Base Intelligence:** Simulated as a random normal distribution variable.
