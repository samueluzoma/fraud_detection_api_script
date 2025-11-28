# =====================================================================
# Flask API for Bank Fraud Detection Model
# =====================================================================

from flask import Flask, request, jsonify          # Flask web framework for API
import joblib                                       # For loading trained model
import numpy as np                                  # For numerical operations
import pandas as pd                                 # For data handling
import os                                           # For file path operations

# Initialize Flask application
app = Flask(__name__)

# =====================================================================
# Load trained model and necessary files at startup
# =====================================================================
try:
    model = joblib.load('best_fraud_model.pkl')           # Loads trained fraud detection model
    feature_columns = joblib.load('feature_columns.pkl')  # Loads feature column names
    label_encoders = joblib.load('label_encoders.pkl')    # Loads label encoders for categorical features
    print("Model and encoders loaded successfully")
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    feature_columns = None
    label_encoders = None

# =====================================================================
# Helper function to preprocess input data
# =====================================================================
def preprocess_input(data):
    """
    Preprocesses incoming data to match model training format
    Args:
        data: Dictionary containing transaction features
    Returns:
        DataFrame ready for model prediction
    """
    # Create DataFrame from input dictionary
    df = pd.DataFrame([data])
    
    # Encode categorical features using saved label encoders
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                # Transform categorical value to numeric using fitted encoder
                df[col] = encoder.transform(df[col])
            except ValueError as e:
                # Handle unknown categories not seen during training
                raise ValueError(f"Unknown category in {col}: {df[col].values[0]}")
    
    # Ensure all required features are present in correct order
    df = df[feature_columns]
    
    return df

# =====================================================================
# Root endpoint - API health check
# =====================================================================
@app.route('/', methods=['GET'])
def home():
    """
    Health check endpoint to verify API is running
    Returns: JSON with API status and available endpoints
    """
    return jsonify({
        'status': 'API is running',
        'model_loaded': model is not None,
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/ (GET)'
        },
        'required_features': feature_columns if feature_columns else []
    })

# =====================================================================
# Prediction endpoint - Main fraud detection API
# =====================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts fraud probability for a transaction
    Expects: JSON with transaction features
    Returns: JSON with fraud prediction and probability
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.'
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        # Validate that data was provided
        if not data:
            return jsonify({
                'error': 'No data provided. Please send JSON data.'
            }), 400
        
        # Validate all required features are present
        missing_features = [f for f in feature_columns if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {missing_features}'
            }), 400
        
        # Preprocess input data
        input_df = preprocess_input(data)
        
        # Make prediction (0 = Not Fraud, 1 = Fraud)
        prediction = model.predict(input_df)[0]
        
        # Get prediction probability if model supports it
        try:
            prediction_proba = model.predict_proba(input_df)[0]
            fraud_probability = float(prediction_proba[1])  # Probability of fraud
        except AttributeError:
            # Some models don't have predict_proba method
            fraud_probability = float(prediction)
        
        # Prepare response
        result = {
            'is_fraud': int(prediction),
            'fraud_probability': round(fraud_probability, 4),
            'message': 'Fraudulent Transaction' if prediction == 1 else 'Legitimate Transaction',
            'risk_level': 'HIGH' if fraud_probability > 0.7 else 'MEDIUM' if fraud_probability > 0.3 else 'LOW'
        }
        
        return jsonify(result), 200
    
    except ValueError as ve:
        # Handle validation errors
        return jsonify({
            'error': str(ve)
        }), 400
    
    except Exception as e:
        # Handle unexpected errors
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

# =====================================================================
# Batch prediction endpoint - For multiple transactions
# =====================================================================
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predicts fraud for multiple transactions at once
    Expects: JSON array of transaction objects
    Returns: JSON array with predictions for each transaction
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.'
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        # Validate that data is a list
        if not isinstance(data, list):
            return jsonify({
                'error': 'Expected a list of transactions'
            }), 400
        
        results = []
        
        # Process each transaction
        for idx, transaction in enumerate(data):
            try:
                # Preprocess input
                input_df = preprocess_input(transaction)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Get probability
                try:
                    prediction_proba = model.predict_proba(input_df)[0]
                    fraud_probability = float(prediction_proba[1])
                except AttributeError:
                    fraud_probability = float(prediction)
                
                # Add result for this transaction
                results.append({
                    'transaction_index': idx,
                    'is_fraud': int(prediction),
                    'fraud_probability': round(fraud_probability, 4),
                    'message': 'Fraudulent Transaction' if prediction == 1 else 'Legitimate Transaction'
                })
            
            except Exception as e:
                # If one transaction fails, include error but continue with others
                results.append({
                    'transaction_index': idx,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': results,
            'total_processed': len(results)
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500

# =====================================================================
# Run the Flask application
# =====================================================================
if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5001))
    
    # Run Flask app
    # debug=False for production, set to True for development
    app.run(host='0.0.0.0', port=port, debug=False)