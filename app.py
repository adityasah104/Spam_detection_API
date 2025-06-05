
# from flask import Flask, request, render_template
# from joblib import load
# from feature_extraction import SMSFeatureExtractor
# from flask import jsonify
# import os

# app = Flask(__name__)
# model = load('spam_model.joblib')
# extractor = SMSFeatureExtractor()  # Initialize the feature extractor

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     message = request.form['message']
#     # Extract features using the extractor instance
#     features = extractor.extract_features(message)
#     prediction = model.predict(features)[0]
    
#     return render_template('index.html', prediction_text=f"Prediction: {prediction}")

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     try:
#         data = request.json
#         message = data['message']
#         features = extractor.extract_features(message)
#         prediction = model.predict(features)[0]
#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# # if __name__ == '__main__':
# #      app.run(debug=True)   /// Changed for Render hosting, For local host keep 36 and 37 line and remove/comment 38,39,40 lines
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(debug=False, host='0.0.0.0', port=port)


# Above code works fine on my local pc

# Below code is for render

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
        prediction = model.predict(features)[0]
        
        # Convert prediction to readable format
        result = "Spam" if prediction == 1 else "Not Spam"
        
        return render_template('index.html', 
                             prediction_text=f"Prediction: {result}",
                             message=message)
    
    except Exception as e:
        logger.error(f"Error in web prediction: {str(e)}")
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
        prediction = model.predict(features)[0]
        
        # Get prediction probability if available
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
        except:
            confidence = None
        
        # Convert prediction to readable format
        result = "spam" if prediction == 1 else "ham"
        
        return jsonify({
            'message': message,
            'prediction': result,
            'prediction_code': int(prediction),
            'confidence': float(confidence) if confidence else None,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error in API prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'prediction': None,
            'status': 'error'
        }), 500

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