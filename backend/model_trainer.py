import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model():
    print("--- Starting IEEE Conference-Grade Model Training ---")

    # --- Step 1: Load Real-World Dataset ---
    try:
        data = pd.read_csv('student-mat.csv', sep=';')
        print("Successfully loaded 'student-mat.csv'.")
    except FileNotFoundError:
        print("ERROR: 'student-mat.csv' not found.")
        print("Please download it from the UCI repository and place it in the backend folder.")
        return

    # --- Step 2: Feature Engineering and Mapping ---
    # We will map the UCI dataset columns to the features our frontend/API expects.
    
    # 1. 'prev_gpa': We'll use G1 and G2 (grades 0-20), average them, and scale to 0-10.
    df = pd.DataFrame()
    df['prev_gpa'] = (data['G1'] + data['G2']) / 4  # (0-20) + (0-20) = 0-40. /4 -> 0-10 scale.

    # 2. 'attendance': We'll use 'absences'. Let's assume a max of 30 absences for a 100% scale.
    # Invert it, so high attendance = good.
    df['attendance'] = ((30 - data['absences']).clip(0, 30) / 30) * 100

    # 3. 'backlogs': Direct map from 'failures' (number of past class failures).
    df['backlogs'] = data['failures']

    # 4. 'mid_term': We'll use G1 (first period grade, 0-20) and scale to 0-100.
    df['mid_term'] = data['G1'] * 5

    # 5. 'engagement': We'll use 'studytime' (1-4 scale). Let's map it to hours/week.
    # 1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs
    study_map = {1: 1, 2: 3, 3: 7, 4: 12}
    df['engagement'] = data['studytime'].map(study_map)

    # 6. 'scholarship': We will use 'paid' (extra paid classes) as a proxy. (yes=1, no=0)
    df['scholarship'] = data['paid'].map({'yes': 1, 'no': 0})

    # 7. 'activities': Direct map from 'activities' (extra-curricular). (yes=1, no=0)
    df['activities'] = data['activities'].map({'yes': 1, 'no': 0})
    
    print("Feature engineering complete. Mapped UCI data to project features.")

    # --- Step 3: Define Target Variable (Dropout) ---
    # This is the core of the research.
    # We define an "at-risk" student as someone who fails the final grade (G3 < 10).
    # 1 = At-Risk (Dropout), 0 = Not At-Risk.
    df['dropout'] = (data['G3'] < 10).astype(int)
    
    print(f"Target variable 'dropout' (G3 < 10) created.")
    print(f"Dataset at-risk rate: {df['dropout'].mean() * 100:.2f}%")

    # --- Step 4: Preprocessing ---
    X = df.drop('dropout', axis=1)
    y = df['dropout']
    
    feature_names = X.columns.tolist() # Save feature names for the scaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data split and scaled: {len(X_train)} train, {len(X_test)} test samples.")

    # --- Step 5: Model Benchmarking ---
    print("--- Starting Model Benchmarking ---")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine (SVC)': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    best_model = None
    best_f1 = -1

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = [accuracy, precision, recall, f1]
        
        # Select the best model based on F1-score (good for imbalanced datasets)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    # --- Step 6: Display Results ---
    print("\n--- Benchmark Results ---")
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    print(results_df.to_markdown(floatfmt=".4f"))
    print(f"\nChampion model selected: {type(best_model).__name__} (F1-Score: {best_f1:.4f})")

    # --- Step 7: Save the Champion Model ---
    joblib.dump(best_model, 'student_dropout_model.pkl')
    # We must also save the scaler, with the correct feature names
    scaler.feature_names_in_ = feature_names
    joblib.dump(scaler, 'scaler.pkl')
    
    print("\nFiles 'student_dropout_model.pkl' and 'scaler.pkl' have been successfully saved.")
    print("Backend Model is ready!")

if __name__ == '__main__':
    train_model()

