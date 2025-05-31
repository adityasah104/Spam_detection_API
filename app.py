
from flask import Flask, request, render_template
from joblib import load
from feature_extraction import SMSFeatureExtractor
from flask import jsonify

app = Flask(__name__)
model = load('spam_model.joblib')
extractor = SMSFeatureExtractor()  # Initialize the feature extractor

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    # Extract features using the extractor instance
    features = extractor.extract_features(message)
    prediction = model.predict(features)[0]
    
    return render_template('index.html', prediction_text=f"Prediction: {prediction}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        message = data['message']
        features = extractor.extract_features(message)
        prediction = model.predict(features)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)