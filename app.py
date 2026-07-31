import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Path to saved model pickle file (if available)
MODEL_FILE = 'model.pkl'

def load_model():
    """Load model from file or build fallback logic if pickle file exists."""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    return None

# Load model into memory on startup
model = load_model()

# Label mapping based on model classes: ["Abnormal", "Inconclusive", "Normal"]
CLASSES = ["Abnormal", "Inconclusive", "Normal"]

@app.route('/')
def home():
    """Render home prediction interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle model prediction requests."""
    try:
        # Expecting feature array from front-end form/JSON request
        data = request.form if request.form else request.get_json()
        
        # Extract features (adjust number of features to match your trained data inputs)
        features = [float(x) for x in data.getlist('features') if x != '']
        
        if not features:
            # If standard key-value attributes are passed
            features = [float(val) for key, val in data.items() if key.startswith('feature')]

        if model is not None:
            # Perform prediction using loaded scikit-learn RandomForest model
            input_array = np.array(features).reshape(1, -1)
            prediction_idx = model.predict(input_array)[0]
            
            # Retrieve class label if prediction is an integer/index
            if isinstance(prediction_idx, (int, np.integer)):
                prediction_label = CLASSES[prediction_idx] if prediction_idx < len(CLASSES) else str(prediction_idx)
            else:
                prediction_label = str(prediction_idx)
                
            probabilities = model.predict_proba(input_array)[0].tolist() if hasattr(model, "predict_proba") else None
        else:
            # Fallback placeholder prediction if model.pkl is missing
            prediction_label = "Inconclusive"
            probabilities = [0.33, 0.34, 0.33]

        return render_template(
            'index.html',
            prediction=prediction_label,
            probabilities=probabilities,
            inputs=features
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
