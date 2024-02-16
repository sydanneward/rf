from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load

app = Flask(__name__)

# Load the pre-trained model
model = load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Initialize list for input features
    input_features = []

    # Define which fields are expected as input
    expected_fields = ['Avg Policy Size', 'Years Positive', 'Operating Acreage', 'Rain Index Variance', 'Distance to Nearest Customer (Miles)']

    # Extract inputs from form
    for field in expected_fields:
        try:
            value = float(request.form[field])
            input_features.append(value)
        except ValueError:
            return jsonify({'error': f'Invalid input for field: {field}'}), 400

    # Reshape input_features for a single sample
    input_features = np.array(input_features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)
    output = prediction[0]

    return jsonify({'prediction': int(output)})

if __name__ == "__main__":
    app.run(debug=True)
