from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import io

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

# --- Load the Pre-trained Model and Scaler ---
try:
    model = joblib.load('student_dropout_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler files not found.")
    print("Please run 'model_trainer.py' first to generate these files.")
    model = None
    scaler = None

# --- Feature Engineering "Recipe" ---
# This function contains the *exact same* logic as your model_trainer.py
# We must apply this to any uploaded file.
def process_dataframe(data):
    try:
        df = pd.DataFrame()
        # 1. 'prev_gpa'
        df['prev_gpa'] = (data['G1'] + data['G2']) / 4
        # 2. 'attendance'
        df['attendance'] = ((30 - data['absences']).clip(0, 30) / 30) * 100
        # 3. 'backlogs'
        df['backlogs'] = data['failures']
        # 4. 'mid_term'
        df['mid_term'] = data['G1'] * 5
        # 5. 'engagement'
        study_map = {1: 1, 2: 3, 3: 7, 4: 12}
        df['engagement'] = data['studytime'].map(study_map)
        # 6. 'scholarship'
        df['scholarship'] = data['paid'].map({'yes': 1, 'no': 0})
        # 7. 'activities'
        df['activities'] = data['activities'].map({'yes': 1, 'no': 0})
        
        # Ensure the column order is identical to what the model was trained on
        feature_order = ['prev_gpa', 'attendance', 'backlogs', 'mid_term', 'engagement', 'scholarship', 'activities']
        
        return df[feature_order]
        
    except KeyError as e:
        print(f"File processing error: Missing required column - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")
        return None

# --- API Endpoints ---

@app.route('/')
def home():
    return "Aura Predict Backend API is running!"

# Define the endpoint for single predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        data = request.get_json()
        print(f"Received data for prediction: {data}")
        feature_order = ['prev_gpa', 'attendance', 'backlogs', 'mid_term', 'engagement', 'scholarship', 'activities']
        ordered_data = {feature: [float(data[feature])] for feature in feature_order}
        input_df = pd.DataFrame(ordered_data)
        input_scaled = scaler.transform(input_df)
        prediction_proba = model.predict_proba(input_scaled)[:, 1]
        risk_probability = prediction_proba[0]

        if risk_probability > 0.65: risk_level = 'High Risk'
        elif risk_probability > 0.35: risk_level = 'Medium Risk'
        else: risk_level = 'Low Risk'

        response = { 'risk_probability': round(risk_probability, 2), 'risk_level': risk_level }
        print(f"Prediction result: {response}")
        return jsonify(response)
    
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An error occurred processing your request.'}), 400

# --- BATCH PREDICTION ENDPOINT (NOW FULLY FUNCTIONAL) ---
@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    # 1. Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        # 2. Read the uploaded file
        # The UCI dataset uses a semicolon separator, so we must specify it.
        raw_data = pd.read_csv(file, sep=';')
        
        # 3. Process the file using the "recipe"
        processed_df = process_dataframe(raw_data.copy())
        
        if processed_df is None:
            return jsonify({'error': 'File format is incorrect. Please ensure it has columns like G1, G2, absences, failures, etc.'}), 400
            
        # 4. Scale the data
        scaled_data = scaler.transform(processed_df)
        
        # 5. Get predictions and probabilities
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)[:, 1] # Probability of class 1 (at-risk)
        
        # 6. Aggregate the results
        total_students = len(processed_df)
        high_risk_count = int(np.sum(predictions))
        average_risk_score = float(np.mean(probabilities))
        
        # 7. Create the final summary
        summary = {
            'total_students': total_students,
            'high_risk_count': high_risk_count,
            'average_risk_score': average_risk_score
        }
        
        print(f"Batch prediction complete: {summary}")
        return jsonify(summary)

    except pd.errors.ParserError:
        return jsonify({'error': 'Failed to parse file. Please ensure it is a valid CSV file with a semicolon (;) separator.'}), 400
    except Exception as e:
        print(f"An error occurred during batch prediction: {e}")
        return jsonify({'error': 'An unexpected error occurred processing your file.'}), 500

# Define a placeholder for the dashboard analytics
@app.route('/dashboard', methods=['GET'])
def dashboard_analytics():
    analytics_data = {
        'risk_distribution': [986, 185, 257], # [Low, Medium, High]
        'factor_importance': {
            'labels': ['Attendance', 'Backlogs', 'GPA', 'Test Scores', 'Engagement'],
            'scores': [92, 85, 75, 60, 45]
        },
        'summary_stats': {
            'overall_risk': 'Medium',
            'high_risk_percentage': 18,
            'top_factor': 'Low Attendance'
        }
    }
    return jsonify(analytics_data)

# API Endpoint for Downloading Batch Report
@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        print("Generating a sample CSV report for download...")
        num_students_report = 1428
        report_data = {
            'student_id': range(1001, 1001 + num_students_report),
            'prev_gpa': np.random.uniform(5.0, 10.0, num_students_report),
            'attendance': np.random.uniform(50, 100, num_students_report),
            'backlogs': np.random.randint(0, 8, num_students_report),
            'risk_probability': np.random.uniform(0.05, 0.95, num_students_report)
        }
        report_df = pd.DataFrame(report_data)
        report_df['risk_probability'] = report_df['risk_probability'].round(2)
        
        def get_risk_level(prob):
            if prob > 0.65: return 'High Risk'
            if prob > 0.35: return 'Medium Risk'
            return 'Low Risk'
            
        report_df['risk_level'] = report_df['risk_probability'].apply(get_risk_level)
        output = io.StringIO()
        report_df.to_csv(output, index=False)
        csv_output = output.getvalue()

        response = make_response(csv_output)
        response.headers["Content-Disposition"] = "attachment; filename=student_risk_report.csv"
        response.headers["Content-type"] = "text/csv"
        
        print("CSV report generated and sent to user.")
        return response

    except Exception as e:
        print(f"An error occurred during report generation: {e}")
        return jsonify({'error': 'An error occurred generating the report.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

    

