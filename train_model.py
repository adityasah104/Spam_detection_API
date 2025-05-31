# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from feature_extraction import SMSFeatureExtractor
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import joblib
# import os

# class SMSSpamModel:
#     def __init__(self):
#         self.model = None
#         self.feature_extractor = SMSFeatureExtractor()
#         self.feature_columns = self.feature_extractor.feature_columns
        
#     def prepare_data(self, df, text_column='Text', label_column='Label'):
#         """Prepare data for training"""
#         print("Extracting features...")
        
#         # Extract features using your existing preprocessing pipeline
#         features_df = self.feature_extractor.extract_batch_features(df[text_column])
        
#         # Combine features with labels
#         X = features_df
#         y = df[label_column].map({'ham': 0, 'spam': 1})  # Convert to binary
        
#         return X, y
    
#     def train(self, df, text_column='Text', label_column='Label', test_size=0.2):
#         """Train the Random Forest model"""
#         X, y = self.prepare_data(df, text_column, label_column)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42, stratify=y
#         )
        
#         print(f"Training data shape: {X_train.shape}")
#         print(f"Testing data shape: {X_test.shape}")
        
#         # Initialize Random Forest with optimized parameters
#         self.model = RandomForestClassifier(
#             n_estimators=100,
#             max_depth=10,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             random_state=42,
#             n_jobs=-1
#         )
        
#         # Train the model
#         print("Training Random Forest model...")
#         self.model.fit(X_train, y_train)
        
#         # Evaluate
#         train_score = self.model.score(X_train, y_train)
#         test_score = self.model.score(X_test, y_test)
        
#         print(f"Training Accuracy: {train_score:.4f}")
#         print(f"Testing Accuracy: {test_score:.4f}")
        
#         # Predictions and detailed evaluation
#         y_pred = self.model.predict(X_test)
        
#         print("\nClassification Report:")
#         print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
#         print("\nConfusion Matrix:")
#         print(confusion_matrix(y_test, y_pred))
        
#         # Feature importance
#         feature_importance = pd.DataFrame({
#             'feature': self.feature_columns,
#             'importance': self.model.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         print("\nTop 10 Most Important Features:")
#         print(feature_importance.head(10))
        
#         return {
#             'train_accuracy': train_score,
#             'test_accuracy': test_score,
#             'feature_importance': feature_importance
#         }
    
#     def predict(self, text):
#         """Predict spam/ham for a single text"""
#         if self.model is None:
#             raise ValueError("Model not trained yet!")
        
#         features = self.feature_extractor.extract_features(text)
#         features_df = pd.DataFrame([features])
        
#         # Ensure all columns are present
#         for col in self.feature_columns:
#             if col not in features_df.columns:
#                 features_df[col] = 0
        
#         # Reorder columns to match training
#         features_df = features_df[self.feature_columns]
        
#         prediction = self.model.predict(features_df)[0]
#         probability = self.model.predict_proba(features_df)[0]
        
#         return {
#             'prediction': 'spam' if prediction == 1 else 'ham',
#             'confidence': max(probability),
#             'spam_probability': probability[1],
#             'ham_probability': probability[0]
#         }
    
#     def save_model(self, model_path='sms_spam_model.pkl'):
#         """Save the trained model"""
#         if self.model is None:
#             raise ValueError("No model to save!")
        
#         model_data = {
#             'model': self.model,
#             'feature_columns': self.feature_columns,
#             'feature_extractor': self.feature_extractor
#         }
        
#         joblib.dump(model_data, model_path)
#         print(f"Model saved to {model_path}")
    
#     def load_model(self, model_path='sms_spam_model.pkl'):
#         """Load a trained model"""
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model file {model_path} not found!")
        
#         model_data = joblib.load(model_path)
#         self.model = model_data['model']
#         self.feature_columns = model_data['feature_columns']
#         self.feature_extractor = model_data['feature_extractor']
        
#         print(f"Model loaded from {model_path}")





# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
from feature_extraction import SMSFeatureExtractor

# Load labeled dataset
df = pd.read_csv('Merged_dataset.csv')  # should contain 'text' and 'label'

# Initialize feature extractor
extractor = SMSFeatureExtractor()

# Feature extraction
features = []
for text in df['text']:
    features.append(extractor.extract_features(text).flatten())  # flatten 2D to 1D

# Create DataFrame with proper column names
X = pd.DataFrame(features, columns=extractor.feature_columns)

# Use 'label' column if it exists, otherwise use 'type'
y = df['label'] if 'label' in df.columns else df['type']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save model
dump(clf, 'spam_model.joblib')
print("Model trained and saved as spam_model.joblib")

# Print accuracy for verification
print(f"Training accuracy: {clf.score(X_train, y_train):.2f}")
print(f"Test accuracy: {clf.score(X_test, y_test):.2f}")