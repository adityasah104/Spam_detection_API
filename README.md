# 📧 Multilingual SMS Spam Detection API

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Active-success)](https://spam-detection-api-rjvc.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready machine learning API for detecting spam in SMS messages across multiple languages. Built with Flask, scikit-learn, and deployed using modern DevOps practices.

## 🎯 Overview

This project implements an end-to-end spam detection system that classifies SMS messages as spam or legitimate (ham) with **97.4% accuracy**. The system is containerized, CI/CD enabled, and deployed on AWS EC2 for high availability.

## ✨ Features

- **High Accuracy**: 97.4% classification accuracy on real-world data
- **Real-time Predictions**: Low-latency API responses with caching mechanisms
- **RESTful API**: Clean, documented endpoints for easy integration
- **Multilingual Support**: Handles SMS in multiple languages
- **Production Ready**: Fully deployed with monitoring and logging
- **DevOps Pipeline**: Automated CI/CD using Jenkins
- **Containerized**: Docker support for consistent deployments
- **Cloud Deployed**: Running on AWS EC2 with optimized configuration

## 🚀 Live Demo

Try the API: [https://spam-detection-api-rjvc.onrender.com/](https://spam-detection-api-rjvc.onrender.com/)

### Quick Test

```bash
curl -X POST https://spam-detection-api-rjvc.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You have won a free iPhone. Click here to claim."}'
```

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Feature Engineering**: TF-IDF Vectorization, NLP preprocessing
- **DevOps**: Docker, Jenkins, AWS EC2
- **Version Control**: Git, GitHub
- **Deployment**: Render/AWS with CI/CD automation

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 97.4% |
| Precision | 96.8% |
| Recall | 95.2% |
| F1-Score | 96.0% |

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│  Flask API   │─────▶│  ML Model   │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │    Cache     │
                     └──────────────┘
```

## 📁 Project Structure

```
Spam_detection_API/
├── app.py                    # Flask application
├── train_model.py            # Model training script
├── feature_extraction.py     # Feature engineering
├── test_api.py              # API testing
├── spam_model.joblib        # Trained model
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
├── Jenkinsfile             # CI/CD pipeline
├── docker-compose.yml      # Multi-container setup
├── .render.yaml            # Render deployment config
└── templates/              # HTML templates
    └── index.html          # API documentation page
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8+
- Docker (optional)
- Git

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/adityasah104/Spam_detection_API.git
cd Spam_detection_API
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model** (optional - pre-trained model included)
```bash
python train_model.py
```

5. **Run the application**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Docker Deployment

1. **Build the image**
```bash
docker build -t spam-detection-api .
```

2. **Run the container**
```bash
docker run -p 5000:5000 spam-detection-api
```

### Docker Compose

```bash
docker-compose up
```

## 📡 API Endpoints

### 1. Home Page
```
GET /
```
Returns the API documentation page.

### 2. Predict Spam
```
POST /predict
Content-Type: application/json

{
  "message": "Your SMS text here"
}
```

**Response:**
```json
{
  "prediction": "spam",
  "confidence": 0.92,
  "message": "Your SMS text here"
}
```

### 3. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🧪 Testing

Run the test suite:
```bash
python test_api.py
```

Test with curl:
```bash
# Test spam message
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "FREE! Win a new car. Call now!"}'

# Test legitimate message
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Hey, are we still meeting for lunch tomorrow?"}'
```

## 🔄 CI/CD Pipeline

The project uses Jenkins for automated CI/CD:

1. **Code Push** → Triggers Jenkins webhook
2. **Build Stage** → Runs tests and builds Docker image
3. **Deploy Stage** → Pushes to container registry
4. **Production** → Auto-deploys to AWS EC2

## ☁️ AWS Deployment

The application is deployed on AWS EC2 with:
- **Auto-scaling**: Handles traffic spikes
- **Load Balancing**: Distributes requests
- **Security Groups**: Restricts access
- **Monitoring**: CloudWatch integration

## 📈 Performance Optimization

- **Caching**: Implemented response caching for frequently tested messages
- **Model Loading**: Lazy loading to reduce startup time
- **Request Batching**: Supports batch predictions
- **Compression**: Gzip compression for API responses

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 👨‍💻 Author

**Aditya Sah**
- GitHub: [@adityasah104](https://github.com/adityasah104)
- LinkedIn: [aditya-sah-574550257](https://www.linkedin.com/in/aditya-sah-574550257/)
- Email: adityasah712@gmail.com

## 🙏 Acknowledgments

- Dataset source: [Spam SMS Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Research Internship at NIT Silchar
- Jenkins community for CI/CD tools

## 📞 Support

For questions or issues, please open an issue on GitHub or contact via email.

---

⭐ If you found this project helpful, please consider giving it a star!
