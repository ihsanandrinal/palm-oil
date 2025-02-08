from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import threading

app = Flask(__name__)

# Load initial model
def load_model():
    global model_data
    model_data = joblib.load('model/model.pkl')

load_model()

@app.route('/')
def home():
    return render_template('index.html', 
                         metrics=model_data['metrics'],
                         historical_data=model_data['historical_data'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_date = pd.to_datetime(request.json['date'])
        days = (input_date - model_data['start_date']).days
        
        if days < 0:
            raise ValueError("Date cannot be before the training data start date")
            
        prediction = model_data['model'].predict([[days]])[0]
        return jsonify({'prediction': round(prediction, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain_model():
    def training_task():
        try:
            # Retrain model
            from model import train_model
            train_model.train_and_save_model()
            # Reload model
            load_model()
        except Exception as e:
            print(f"Retraining failed: {str(e)}")

    # Start retraining in background thread
    if not threading.active_count() > 2:  # Simple way to prevent multiple concurrent retrains
        thread = threading.Thread(target=training_task)
        thread.start()
        return jsonify({'status': 'Retraining started'})
    return jsonify({'error': 'Retraining already in progress'}), 429

if __name__ == '__main__':
    app.run(debug=True)