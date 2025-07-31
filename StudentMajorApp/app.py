from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the saved model and necessary encoders/scalers
model = joblib.load('models/student_major_predictor_model.pkl')
scaler = joblib.load('models/student_major_scaler.pkl')
target_encoder = joblib.load('models/student_major_target_encoder.pkl')

app = Flask(__name__, template_folder='.')  # Set template folder to current directory

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from the user
        input_data = request.json

        # Extract features in the expected order
        features = [
            input_data['High School GPA'],
            input_data['Entrance Exam Score'],
            input_data['Gender'],
            input_data['Learning Style'],
            input_data['Study Environment']
        ]

        # Preprocess the features
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        # Predict the major
        prediction = model.predict(features)
        major = target_encoder.inverse_transform(prediction)[0]

        return jsonify({"predicted_major": major})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)