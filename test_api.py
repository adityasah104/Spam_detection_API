import requests
import json

class APITester:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
    
    def test_health(self):
        """Test health endpoint"""
        try:
            response = requests.get(f'{self.base_url}/health')
            print(f"Health Check: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def test_single_prediction(self, text):
        """Test single text prediction"""
        try:
            data = {'text': text}
            response = requests.post(f'{self.base_url}/predict', json=data)
            
            print(f"\nPrediction Test:")
            print(f"Input: {text}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Spam Probability: {result['spam_probability']:.2%}")
                print(f"Ham Probability: {result['ham_probability']:.2%}")
            else:
                print(f"Error: {response.json()}")
            
            return response.status_code == 200
        except Exception as e:
            print(f"Single prediction test failed: {e}")
            return False
    
    def test_batch_prediction(self, texts):
        """Test batch prediction"""
        try:
            data = {'texts': texts}
            response = requests.post(f'{self.base_url}/predict_batch', json=data)
            
            print(f"\nBatch Prediction Test:")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Total Processed: {result['total_processed']}")
                
                for res in result['results']:
                    if 'error' not in res:
                        print(f"Text: {res['input_text'][:50]}...")
                        print(f"Prediction: {res['prediction']} ({res['confidence']:.2%})")
                    else:
                        print(f"Error for text {res['index']}: {res['error']}")
            else:
                print(f"Error: {response.json()}")
            
            return response.status_code == 200
        except Exception as e:
            print(f"Batch prediction test failed: {e}")
            return False
    
    def test_model_info(self):
        """Test model info endpoint"""
        try:
            response = requests.get(f'{self.base_url}/model_info')
            print(f"\nModel Info Test:")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                info = response.json()
                print(f"Model Type: {info['model_type']}")
                print(f"Features Count: {info['features_count']}")
                print(f"Supported Languages: {info['supported_languages']}")
            else:
                print(f"Error: {response.json()}")
            
            return response.status_code == 200
        except Exception as e:
            print(f"Model info test failed: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("Starting comprehensive API testing...")
        
        # Test samples in different languages
        test_texts = [
            # English spam examples
            "CONGRATULATIONS! You have won $1000! Call now to claim your prize!",
            "FREE! Click here to get your reward now! Limited time offer!",
            
            # English ham examples
            "Hi, how are you doing today? Let's meet for coffee.",
            "The meeting is scheduled for tomorrow at 3 PM.",
            
            # Assamese examples (you can add more)
            "বিনামূলীয়া উপহাৰ! এতিয়াই ক্লিক কৰক!",  # Spam-like
            "আজি কেনেকুৱা আছা? আহক আড্ডা দিওঁ।",  # Ham-like
            
            # Bengali examples (you can add more)
            "বিনামূল্যে পুরস্কার! এখনই ক্লিক করুন!",  # Spam-like
            "আজ কেমন আছো? চলো আড্ডা দেওয়া যাক।"  # Ham-like
        ]
        
        # Run tests
        results = []
        
        # Health check
        results.append(("Health Check", self.test_health()))
        
        # Model info
        results.append(("Model Info", self.test_model_info()))
        
        # Single predictions
        for i, text in enumerate(test_texts[:4]):  # Test first 4 individually
            results.append((f"Single Prediction {i+1}", self.test_single_prediction(text)))
        
        # Batch prediction
        results.append(("Batch Prediction", self.test_batch_prediction(test_texts)))
        
        # Summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        for test_name, success in results:
            status = "PASS" if success else "FAIL"
            print(f"{test_name}: {status}")
        
        total_tests = len(results)
        passed_tests = sum(1 for _, success in results if success)
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")

if __name__ == "__main__":
    # Initialize tester
    tester = APITester()
    
    # Run comprehensive tests
    tester.run_comprehensive_test()