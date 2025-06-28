from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from joblib import load
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for API endpoints

# Global variables for model and extractor
model = None
extractor = None

def load_model_and_extractor():
    """Load model and feature extractor once during startup"""
    global model, extractor
    try:
        # Load the spam detection model
        model_path = os.path.join(os.path.dirname(__file__), 'spam_model.joblib')
        if os.path.exists(model_path):
            model = load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Import and initialize feature extractor
        from feature_extraction import SMSFeatureExtractor
        extractor = SMSFeatureExtractor()
        logger.info("Feature extractor initialized successfully")
        
    except Exception as e:
        logger.error(f"Error loading model or extractor: {str(e)}")
        raise e

def normalize_prediction(prediction):
    """
    Normalize prediction to standard format regardless of model output type
    Returns: (normalized_string, prediction_code)
    """
    # Convert prediction to string and normalize
    pred_str = str(prediction).lower().strip()
    
    # Handle different possible outputs
    if pred_str in ['spam', '1', 1]:
        return 'spam', 1
    elif pred_str in ['ham', 'not spam', 'notspam', '0', 0]:
        return 'ham', 0
    else:
        # Log unexpected prediction format
        logger.warning(f"Unexpected prediction format: {prediction}")
        # Default to ham for safety
        return 'ham', 0

# Load model and extractor when the app starts
try:
    load_model_and_extractor()
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}")
    # In production, you might want to exit here
    # sys.exit(1)

@app.route('/')
def home():
    """Home page route"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return jsonify({'error': 'Template not found or error in rendering'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'message': 'SMS Spam Detection API is running',
        'model_loaded': model is not None,
        'extractor_loaded': extractor is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Web form prediction route"""
    try:
        if model is None or extractor is None:
            return render_template('index.html', 
                                 prediction_text="Error: Model or extractor not loaded")
        
        message = request.form.get('message', '').strip()
        if not message:
            return render_template('index.html', 
                                 prediction_text="Error: Please enter a message")
        
        # Extract features and make prediction
        features = extractor.extract_features(message)
        raw_prediction = model.predict(features)[0]
        
        # Normalize prediction
        result, prediction_code = normalize_prediction(raw_prediction)
        
        # Get confidence score if available
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities) * 100
            confidence_text = f" (Confidence: {confidence:.1f}%)"
        except Exception as e:
            logger.info(f"Confidence not available: {e}")
            confidence_text = ""
        
        # Convert to display format
        display_result = "Spam" if result == "spam" else "Not Spam"
        
        return render_template('index.html', 
                             prediction_text=f"Prediction: {display_result}{confidence_text}",
                             message=message)
    
    except Exception as e:
        logger.error(f"Error in web prediction: {str(e)}")
        return render_template('index.html', 
                             prediction_text=f"Error: {str(e)}")

@app.route('/predict_simple', methods=['POST'])
def predict_simple_web():
    """Simple web form prediction route"""
    try:
        if model is None or extractor is None:
            return render_template('index.html', 
                                 prediction_text="Error: Model or extractor not loaded")
        
        message = request.form.get('message', '').strip()
        if not message:
            return render_template('index.html', 
                                 prediction_text="Error: Please enter a message")
        
        # Extract features and make prediction
        features = extractor.extract_features(message)
        raw_prediction = model.predict(features)[0]
        
        # Normalize prediction
        result, prediction_code = normalize_prediction(raw_prediction)
        
        # Convert to display format
        display_result = "Spam" if result == "spam" else "Not Spam"
        
        return render_template('index.html', 
                             prediction_text=f"{display_result}",
                             message=message)
    
    except Exception as e:
        logger.error(f"Error in simple web prediction: {str(e)}")
        return render_template('index.html', 
                             prediction_text=f"Error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        # Check if model and extractor are loaded
        if model is None or extractor is None:
            return jsonify({
                'error': 'Model or feature extractor not loaded',
                'prediction': None
            }), 500
        
        # Get JSON data
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing message in request body',
                'prediction': None
            }), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({
                'error': 'Message cannot be empty',
                'prediction': None
            }), 400
        
        # Extract features and make prediction
        features = extractor.extract_features(message)
        raw_prediction = model.predict(features)[0]
        
        # Log the raw prediction for debugging
        logger.info(f"Raw prediction from model: {raw_prediction} (type: {type(raw_prediction)})")
        
        # Normalize prediction
        result, prediction_code = normalize_prediction(raw_prediction)
        
        # Get prediction probability if available
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            # Try to get individual class probabilities
            if len(probabilities) == 2:
                # Assuming index 0 = ham, index 1 = spam (common convention)
                if hasattr(model, 'classes_'):
                    classes = model.classes_
                    if len(classes) == 2:
                        class_probs = {}
                        for i, class_label in enumerate(classes):
                            normalized_class, _ = normalize_prediction(class_label)
                            class_probs[normalized_class] = float(probabilities[i])
                    else:
                        class_probs = {'ham': float(probabilities[0]), 'spam': float(probabilities[1])}
                else:
                    class_probs = {'ham': float(probabilities[0]), 'spam': float(probabilities[1])}
            else:
                class_probs = None
                
        except Exception as e:
            logger.info(f"Probability calculation failed: {e}")
            confidence = None
            class_probs = None
        
        response_data = {
            'message': message,
            'prediction': result,
            'prediction_code': prediction_code,
            'confidence': float(confidence) if confidence else None,
            'raw_prediction': str(raw_prediction),  # For debugging
            'status': 'success'
        }
        
        # Add class probabilities if available
        if class_probs:
            response_data['probabilities'] = class_probs
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in API prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'prediction': None,
            'status': 'error'
        }), 500

@app.route('/api/predict_simple', methods=['POST'])
def api_predict_simple():
    """Simple API endpoint that returns only the prediction"""
    try:
        # Check if model and extractor are loaded
        if model is None or extractor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message'}), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Extract features and make prediction
        features = extractor.extract_features(message)
        raw_prediction = model.predict(features)[0]
        
        # Normalize prediction
        result, _ = normalize_prediction(raw_prediction)
        
        # Return only the prediction
        return jsonify({'prediction': result})
    
    except Exception as e:
        logger.error(f"Error in simple prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        info = {
            'model_type': str(type(model).__name__),
            'model_loaded': True
        }
        
        # Try to get model classes if available
        if hasattr(model, 'classes_'):
            info['classes'] = [str(c) for c in model.classes_]
        
        # Try to get feature names if available
        if hasattr(model, 'feature_names_in_'):
            info['feature_count'] = len(model.feature_names_in_)
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)