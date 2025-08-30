#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for HalluciNet Detector
Tests all endpoints including upload, EDA, training, and prediction functionality
"""

import requests
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

class HalluciNetAPITester:
    def __init__(self, base_url="https://deepfact-check.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.uploaded_datasets = []
        self.eda_results = {}
        self.training_started = False

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED {details}")
        else:
            print(f"❌ {name} - FAILED {details}")
        return success

    def test_api_health(self):
        """Test basic API connectivity"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Message: {data.get('message', 'N/A')}"
            return self.log_test("API Health Check", success, details)
        except Exception as e:
            return self.log_test("API Health Check", False, f"Error: {str(e)}")

    def test_upload_dataset(self):
        """Test dataset upload functionality"""
        try:
            # Use the validation dataset
            dataset_path = "/app/backend/datasets/val_agnostic.csv"
            
            if not os.path.exists(dataset_path):
                return self.log_test("Dataset Upload", False, "Sample dataset not found")
            
            with open(dataset_path, 'rb') as f:
                files = {'file': ('val_agnostic.csv', f, 'text/csv')}
                response = requests.post(f"{self.api_url}/upload-dataset", files=files, timeout=30)
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                self.uploaded_datasets.append(data)
                details += f", Dataset ID: {data.get('id')}, Rows: {data.get('rows')}, Columns: {len(data.get('columns', []))}"
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("Dataset Upload", success, details)
        except Exception as e:
            return self.log_test("Dataset Upload", False, f"Error: {str(e)}")

    def test_get_datasets(self):
        """Test retrieving uploaded datasets"""
        try:
            response = requests.get(f"{self.api_url}/datasets", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                datasets = response.json()
                details += f", Found {len(datasets)} datasets"
                # Update our dataset list
                if datasets:
                    self.uploaded_datasets = datasets
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("Get Datasets", success, details)
        except Exception as e:
            return self.log_test("Get Datasets", False, f"Error: {str(e)}")

    def test_dataset_preview(self):
        """Test dataset preview functionality"""
        if not self.uploaded_datasets:
            return self.log_test("Dataset Preview", False, "No datasets available")
        
        try:
            dataset_id = self.uploaded_datasets[0]['id']
            response = requests.get(f"{self.api_url}/dataset/{dataset_id}?limit=5", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Rows: {data.get('rows')}, Columns: {len(data.get('columns', []))}, Has Labels: {data.get('has_labels')}"
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("Dataset Preview", success, details)
        except Exception as e:
            return self.log_test("Dataset Preview", False, f"Error: {str(e)}")

    def test_eda_analysis(self):
        """Test Exploratory Data Analysis"""
        if not self.uploaded_datasets:
            return self.log_test("EDA Analysis", False, "No datasets available")
        
        try:
            dataset_id = self.uploaded_datasets[0]['id']
            response = requests.post(f"{self.api_url}/eda/{dataset_id}", timeout=60)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                self.eda_results[dataset_id] = data
                
                # Check EDA components
                has_class_dist = bool(data.get('class_distribution'))
                has_type_dist = bool(data.get('type_distribution'))
                has_length_stats = bool(data.get('length_stats'))
                has_visualizations = bool(data.get('visualizations'))
                
                details += f", Class Dist: {has_class_dist}, Type Dist: {has_type_dist}, Length Stats: {has_length_stats}, Visualizations: {len(data.get('visualizations', []))}"
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("EDA Analysis", success, details)
        except Exception as e:
            return self.log_test("EDA Analysis", False, f"Error: {str(e)}")

    def test_model_stats(self):
        """Test model statistics endpoint"""
        try:
            response = requests.get(f"{self.api_url}/model-stats", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Models Trained: {data.get('models_trained')}, Model Ready: {data.get('model_ready')}"
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("Model Stats", success, details)
        except Exception as e:
            return self.log_test("Model Stats", False, f"Error: {str(e)}")

    def test_start_training(self):
        """Test model training initiation"""
        if not self.uploaded_datasets:
            return self.log_test("Start Training", False, "No datasets available")
        
        # Check if we have labeled datasets
        labeled_datasets = [d for d in self.uploaded_datasets if 'label' in d.get('columns', [])]
        if not labeled_datasets:
            return self.log_test("Start Training", False, "No labeled datasets available")
        
        try:
            dataset_ids = [d['id'] for d in labeled_datasets]
            payload = {
                "dataset_ids": dataset_ids,
                "n_models": 5,
                "test_split": 0.2,
                "random_seed": 42
            }
            
            response = requests.post(f"{self.api_url}/train-models", json=payload, timeout=30)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                self.training_started = True
                details += f", Message: {data.get('message')}"
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("Start Training", success, details)
        except Exception as e:
            return self.log_test("Start Training", False, f"Error: {str(e)}")

    def test_training_status(self):
        """Test training status monitoring"""
        if not self.training_started:
            return self.log_test("Training Status", False, "Training not started")
        
        try:
            # Check status multiple times to see progress
            for i in range(3):
                response = requests.get(f"{self.api_url}/training-status", timeout=10)
                success = response.status_code == 200
                
                if success:
                    data = response.json()
                    status = data.get('status')
                    progress = data.get('progress', 0)
                    message = data.get('message', '')
                    
                    print(f"   Training Status Check {i+1}: {status} ({progress}%) - {message}")
                    
                    if status in ['completed', 'error']:
                        break
                    elif status in ['training', 'processing']:
                        time.sleep(5)  # Wait before next check
                else:
                    break
            
            details = f"Status: {response.status_code}"
            if success:
                details += f", Final Status: {data.get('status')}, Progress: {data.get('progress')}%"
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("Training Status", success, details)
        except Exception as e:
            return self.log_test("Training Status", False, f"Error: {str(e)}")

    def test_single_prediction(self):
        """Test single hallucination prediction"""
        try:
            # Sample prediction data from the review request
            payload = {
                "hyp": "Paris is the capital of Germany",
                "tgt": "Paris is the capital of France",
                "src": "What is the capital of France?",
                "task": "DM"
            }
            
            response = requests.post(f"{self.api_url}/predict", json=payload, timeout=30)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Predicted Type: {data.get('predicted_type')}, Severity: {data.get('severity_percent')}%"
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("Single Prediction", success, details)
        except Exception as e:
            return self.log_test("Single Prediction", False, f"Error: {str(e)}")

    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        if not self.uploaded_datasets:
            return self.log_test("Batch Prediction", False, "No datasets available")
        
        try:
            dataset_id = self.uploaded_datasets[0]['id']
            response = requests.post(f"{self.api_url}/batch-predict/{dataset_id}", timeout=120)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Total Predictions: {data.get('total_predictions')}, Download Path: {data.get('download_path')}"
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("Batch Prediction", success, details)
        except Exception as e:
            return self.log_test("Batch Prediction", False, f"Error: {str(e)}")

    def test_download_results(self):
        """Test results download functionality"""
        if not self.uploaded_datasets:
            return self.log_test("Download Results", False, "No datasets available")
        
        try:
            dataset_id = self.uploaded_datasets[0]['id']
            response = requests.get(f"{self.api_url}/download-results/{dataset_id}", timeout=30)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                content_type = response.headers.get('content-type', '')
                content_length = len(response.content)
                details += f", Content-Type: {content_type}, Size: {content_length} bytes"
            else:
                details += f", Error: {response.text}"
                
            return self.log_test("Download Results", success, details)
        except Exception as e:
            return self.log_test("Download Results", False, f"Error: {str(e)}")

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🧪 Starting HalluciNet Detector API Tests")
        print("=" * 60)
        
        # Basic connectivity
        if not self.test_api_health():
            print("❌ API is not accessible. Stopping tests.")
            return False
        
        # Data management tests
        self.test_upload_dataset()
        self.test_get_datasets()
        self.test_dataset_preview()
        
        # Analysis tests
        self.test_eda_analysis()
        self.test_model_stats()
        
        # Training tests (may take time)
        print("\n🔄 Starting model training tests (this may take several minutes)...")
        self.test_start_training()
        self.test_training_status()
        
        # Wait a bit more for training to potentially complete
        if self.training_started:
            print("⏳ Waiting for training to progress...")
            time.sleep(10)
            self.test_training_status()
        
        # Prediction tests
        print("\n🔮 Testing prediction functionality...")
        self.test_single_prediction()
        self.test_batch_prediction()
        self.test_download_results()
        
        # Final summary
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return True
        else:
            failed_tests = self.tests_run - self.tests_passed
            print(f"⚠️  {failed_tests} test(s) failed")
            return False

def main():
    """Main test execution"""
    print(f"🚀 HalluciNet Detector Backend Testing")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tester = HalluciNetAPITester()
    success = tester.run_all_tests()
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0 if success else 1



if __name__ == "__main__":
    sys.exit(main())