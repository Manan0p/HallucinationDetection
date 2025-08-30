from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import base64
import csv
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import json
import io
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import nltk
import textstat
from collections import Counter
import warnings
import shutil
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()
DATASETS = []

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for storing models and data
trained_models = []
feature_vectorizer = None
processed_data = {}
training_status = {"status": "idle", "progress": 0, "message": ""}

# Pydantic Models
class DatasetInfo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    size: int
    rows: int
    columns: List[str]
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False

class EDAResults(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str
    class_distribution: Dict[str, int]
    type_distribution: Dict[str, int] 
    length_stats: Dict[str, Any]
    similarity_stats: Dict[str, Any]
    visualizations: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ModelTrainingRequest(BaseModel):
    dataset_ids: List[str]
    n_models: int = 5
    test_split: float = 0.2
    random_seed: int = 42

class PredictionRequest(BaseModel):
    hyp: str
    tgt: str
    src: str
    task: str

class PredictionResult(BaseModel):
    id: str
    predicted_type: str
    severity_percent: int

class TrainingStatus(BaseModel):
    status: str
    progress: int
    message: str
    models_trained: int = 0
    current_accuracy: float = 0.0

# Data Processing Functions
def clean_text(text):
    """Clean and normalize text data"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove <define>...</define> tags
    text = re.sub(r'<define>(.*?)</define>', r'\1', text)
    
    # Remove other HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Keep original case for now (some tasks might need it)
    return text.strip()

def extract_features(hyp, tgt, src, task):
    """Extract comprehensive features for the model"""
    features = {}
    
    # Clean texts
    hyp_clean = clean_text(str(hyp))
    tgt_clean = clean_text(str(tgt))
    src_clean = clean_text(str(src))
    
    # Length features
    features['hyp_length'] = len(hyp_clean)
    features['tgt_length'] = len(tgt_clean)
    features['src_length'] = len(src_clean)
    features['length_ratio_hyp_tgt'] = len(hyp_clean) / max(len(tgt_clean), 1)
    features['length_ratio_hyp_src'] = len(hyp_clean) / max(len(src_clean), 1)
    
    # Word count features
    hyp_words = hyp_clean.split()
    tgt_words = tgt_clean.split()
    src_words = src_clean.split()
    
    features['hyp_word_count'] = len(hyp_words)
    features['tgt_word_count'] = len(tgt_words)
    features['src_word_count'] = len(src_words)
    
    # Text similarity features
    def jaccard_similarity(text1, text2):
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) > 0 else 0
    
    features['jaccard_hyp_tgt'] = jaccard_similarity(hyp_clean, tgt_clean)
    features['jaccard_hyp_src'] = jaccard_similarity(hyp_clean, src_clean)
    
    # Readability features (handle NaN/inf values)
    try:
        hyp_readability = textstat.flesch_reading_ease(hyp_clean)
        features['hyp_readability'] = hyp_readability if np.isfinite(hyp_readability) else 0.0
    except:
        features['hyp_readability'] = 0.0
    
    try:
        tgt_readability = textstat.flesch_reading_ease(tgt_clean)
        features['tgt_readability'] = tgt_readability if np.isfinite(tgt_readability) else 0.0
    except:
        features['tgt_readability'] = 0.0
    
    # Task encoding (one-hot)
    features['task_DM'] = 1 if task == 'DM' else 0
    features['task_MT'] = 1 if task == 'MT' else 0
    features['task_PG'] = 1 if task == 'PG' else 0
    
    return features

def create_feature_matrix(df, vectorizer=None, fit=True):
    """Create comprehensive feature matrix"""
    # Extract hand-crafted features
    feature_rows = []
    for _, row in df.iterrows():
        features = extract_features(row['hyp'], row['tgt'], row['src'], row['task'])
        feature_rows.append(features)
    
    feature_df = pd.DataFrame(feature_rows)
    
    # TF-IDF features on combined text
    combined_texts = []
    for _, row in df.iterrows():
        combined = f"{clean_text(str(row['hyp']))} {clean_text(str(row['tgt']))}"
        combined_texts.append(combined)
    
    if fit and vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        tfidf_features = vectorizer.fit_transform(combined_texts).toarray()
    else:
        tfidf_features = vectorizer.transform(combined_texts).toarray()
    
    # Combine features
    tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
    
    # Reset indices and concatenate
    feature_df.reset_index(drop=True, inplace=True)
    tfidf_df.reset_index(drop=True, inplace=True)
    
    final_features = pd.concat([feature_df, tfidf_df], axis=1)
    
    return final_features.values, vectorizer

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def create_hallucination_model():
    """Create the deep learning model architecture"""
    model = Sequential([
        Dense(100, activation='relu', input_dim=None),  # Will be set dynamically
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(5, activation='softmax')  # 5 classes: 0=No, 1=Factual, 2=Contextual, 3=Logical, 4=Multimodal
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_labels(labels_series):
    """Convert text labels to numeric"""
    label_map = {
        'Not Hallucination': 0,
        'No Hallucination': 0,
        'Hallucination': 1,  # Add this mapping for binary datasets
        'Factual Hallucination': 1,
        'Contextual Hallucination': 2,
        'Logical Hallucination': 3,
        'Multimodal Hallucination': 4
    }
    
    return labels_series.map(label_map).fillna(0).astype(int)

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Hallucination Detection System API"}

@api_router.post("/upload-dataset", response_model=DatasetInfo)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and process a dataset"""
    try:
        # Read the file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Create dataset info
        dataset_info = DatasetInfo(
            filename=file.filename,
            size=len(content),
            rows=len(df),
            columns=list(df.columns)
        )
        
        # Store in database and local cache
        await db.datasets.insert_one(dataset_info.dict())
        processed_data[dataset_info.id] = df
        
        return dataset_info
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@api_router.get("/datasets", response_model=List[DatasetInfo])
async def get_datasets():
    """Get all uploaded datasets"""
    datasets = await db.datasets.find().to_list(1000)
    return [DatasetInfo(**dataset) for dataset in datasets]

@api_router.get("/dataset/{dataset_id}")
async def get_dataset_preview(dataset_id: str, limit: int = 10):
    """Get preview of a dataset"""
    # Try to reload dataset if not in memory
    if dataset_id not in processed_data:
        # Check if dataset exists in database
        dataset_doc = await db.datasets.find_one({"id": dataset_id})
        if not dataset_doc:
            raise HTTPException(status_code=404, detail="Dataset not found in database")
        
        # Try to reload from file if it exists
        dataset_path = f"/app/backend/datasets/{dataset_doc['filename']}"
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                processed_data[dataset_id] = df
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Could not reload dataset: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail="Dataset file not found")
    
    df = processed_data[dataset_id]
    
    # Clean the dataframe to handle NaN and infinite values
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'float32']:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    preview = df_clean.head(limit).to_dict('records')
    
    info = {
        'rows': len(df),
        'columns': list(df.columns),
        'preview': preview,
        'has_labels': 'label' in df.columns,
        'has_hallucination_prob': 'p(Hallucination)' in df.columns
    }
    
    return info

@api_router.post("/eda/{dataset_id}")
async def perform_eda(dataset_id: str):
    """Perform Exploratory Data Analysis on a dataset"""
    # Try to reload dataset if not in memory
    if dataset_id not in processed_data:
        # Check if dataset exists in database
        dataset_doc = await db.datasets.find_one({"id": dataset_id})
        if not dataset_doc:
            raise HTTPException(status_code=404, detail="Dataset not found in database")
        
        # Try to reload from file if it exists
        dataset_path = f"/app/backend/datasets/{dataset_doc['filename']}"
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                processed_data[dataset_id] = df
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Could not reload dataset: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail="Dataset file not found")
    
    df = processed_data[dataset_id].copy()
    
    # Initialize results
    eda_results = {
        'class_distribution': {},
        'type_distribution': {},
        'length_stats': {},
        'similarity_stats': {},
        'visualizations': []
    }
    
    # Class distribution
    if 'label' in df.columns:
        class_dist = df['label'].value_counts().to_dict()
        eda_results['class_distribution'] = class_dist
    
    # Task distribution 
    if 'task' in df.columns:
        task_dist = df['task'].value_counts().to_dict()
        eda_results['type_distribution'] = task_dist
    
    # Length statistics
    if 'hyp' in df.columns and 'tgt' in df.columns:
        df['hyp_length'] = df['hyp'].astype(str).str.len()
        df['tgt_length'] = df['tgt'].astype(str).str.len()
        
        length_stats = {
            'hyp_avg_length': float(df['hyp_length'].mean()),
            'tgt_avg_length': float(df['tgt_length'].mean()),
            'hyp_std_length': float(df['hyp_length'].std()),
            'tgt_std_length': float(df['tgt_length'].std())
        }
        eda_results['length_stats'] = length_stats
    
    # Create visualizations
    plt.style.use('default')
    
    # Class distribution plot
    if 'label' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        df['label'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Distribution of Hallucination Labels', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save as base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        eda_results['visualizations'].append(f"data:image/png;base64,{img_base64}")
        plt.close()
    
    # Task distribution plot
    if 'task' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        df['task'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
        ax.set_title('Task Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('')
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        eda_results['visualizations'].append(f"data:image/png;base64,{img_base64}")
        plt.close()
    
    # Length distribution plot
    if 'hyp_length' in df.columns and 'tgt_length' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.hist(df['hyp_length'], bins=50, alpha=0.7, color='lightcoral', label='Hypothesis')
        ax1.hist(df['tgt_length'], bins=50, alpha=0.7, color='lightblue', label='Target')
        ax1.set_title('Text Length Distribution')
        ax1.set_xlabel('Character Count')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        ax2.scatter(df['hyp_length'], df['tgt_length'], alpha=0.6, color='green')
        ax2.set_title('Hypothesis vs Target Length')
        ax2.set_xlabel('Hypothesis Length')
        ax2.set_ylabel('Target Length')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        eda_results['visualizations'].append(f"data:image/png;base64,{img_base64}")
        plt.close()
    
    # Store results
    eda_doc = EDAResults(
        dataset_id=dataset_id,
        **eda_results
    )
    await db.eda_results.insert_one(eda_doc.dict())
    
    return eda_results

@api_router.post("/train-models")
async def train_models(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train ensemble of hallucination detection models"""
    global training_status, trained_models, feature_vectorizer
    
    training_status = {"status": "starting", "progress": 0, "message": "Initializing training..."}
    background_tasks.add_task(train_models_background, request)
    
    return {"message": "Training started", "status": training_status}

async def train_models_background(request: ModelTrainingRequest):
    """Background task to train models"""
    global training_status, trained_models, feature_vectorizer
    
    try:
        # Combine datasets
        all_data = []
        logger.info("Starting model training...")
        logger.info(f"Number of models before training: {len(trained_models)}")

        for dataset_id in request.dataset_ids:
            if dataset_id not in processed_data:
                # Try to reload dataset from database
                dataset_doc = await db.datasets.find_one({"id": dataset_id})
                if dataset_doc:
                    dataset_path = f"/app/backend/datasets/{dataset_doc['filename']}"
                    if os.path.exists(dataset_path):
                        try:
                            df = pd.read_csv(dataset_path)
                            processed_data[dataset_id] = df
                        except Exception as e:
                            continue
            if dataset_id in processed_data:
                df = processed_data[dataset_id].copy()
                if 'task' in df.columns:  # Only use labeled data for training
                    all_data.append(df)

        if not all_data:
            training_status["status"] = "error"
            training_status["progress"] = 0
            training_status["message"] = f"No labeled datasets found. Available: {list(processed_data.keys())}, Requested: {request.dataset_ids}"
            return

        combined_df = pd.concat(all_data, ignore_index=True)

        # Prepare features and labels
        training_status["status"] = "processing"
        training_status["progress"] = 10
        training_status["message"] = "Extracting features..."

        X, feature_vectorizer = create_feature_matrix(combined_df, fit=True)
        y = prepare_labels(combined_df['task'])

        # Convert to numpy arrays to ensure consistent indexing
        X = np.array(X)
        y = np.array(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_split, random_state=request.random_seed, stratify=y
        )

        # Train ensemble
        trained_models.clear()
        training_status["status"] = "training"
        training_status["progress"] = 20
        training_status["message"] = f"Training {request.n_models} models..."

        for i in range(request.n_models):
            # Bootstrap sampling
            n_samples = len(X_train)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Use numpy indexing for arrays
            X_bootstrap = X_train[bootstrap_indices]
            y_bootstrap = y_train[bootstrap_indices]

            # Create and train model
            model = create_hallucination_model()

            # Set input dimension
            model.layers[0].build((None, X.shape[1]))
            model.layers[0].input_spec.axes = {1: X.shape[1]}

            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )

            # Train
            history = model.fit(
                X_bootstrap, y_bootstrap,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )

            trained_models.append(model)

            progress = 20 + int((i + 1) * 60 / request.n_models)
            training_status["progress"] = progress
            training_status["message"] = f"Trained model {i+1}/{request.n_models}"
            logger.info(f"Model {i+1}/{request.n_models} trained. Progress: {progress}%.")

        # Evaluate ensemble
        training_status["status"] = "evaluating"
        training_status["progress"] = 90
        training_status["message"] = "Evaluating ensemble..."

        ensemble_predictions = []
        for model in trained_models:
            pred = model.predict(X_test, verbose=0)
            ensemble_predictions.append(np.argmax(pred, axis=1))

        ensemble_predictions = np.array(ensemble_predictions)
        final_predictions = []

        for i in range(len(X_test)):
            votes = ensemble_predictions[:, i]
            # Majority vote
            final_pred = np.bincount(votes).argmax()
            final_predictions.append(final_pred)

        accuracy = accuracy_score(y_test, final_predictions)

        training_status["status"] = "completed"
        training_status["progress"] = 100
        training_status["message"] = f"Training completed. Ensemble accuracy: {accuracy:.3f}"
        training_status["models_trained"] = len(trained_models)
        training_status["current_accuracy"] = accuracy
        logger.info(f"Training completed. Number of models after training: {len(trained_models)}")

    except Exception as e:
        training_status["status"] = "error"
        training_status["progress"] = 0
        training_status["message"] = f"Training failed: {str(e)}"

@api_router.get("/training-status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    return TrainingStatus(**training_status)

@api_router.post("/predict", response_model=PredictionResult)
async def predict_hallucination(request: PredictionRequest):
    """Predict hallucination for a single instance"""
    global trained_models, feature_vectorizer
    
    if not trained_models or feature_vectorizer is None:
        raise HTTPException(status_code=400, detail="No trained models available. Please train models first.")
    
    try:
        # Create temporary dataframe
        temp_df = pd.DataFrame([{
            'hyp': request.hyp,
            'tgt': request.tgt,
            'src': request.src,
            'task': request.task
        }])
        
        # Extract features
        X, _ = create_feature_matrix(temp_df, feature_vectorizer, fit=False)
        
        # Get predictions from all models
        predictions = []
        for model in trained_models:
            pred = model.predict(X, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            predictions.append(pred_class)
        
        # Ensemble logic
        predictions = np.array(predictions)
        hallucination_votes = np.sum(predictions > 0)  # Count non-zero (hallucination) votes
        severity_percent = int((hallucination_votes / len(trained_models)) * 100)
        
        if hallucination_votes == 0:
            predicted_type = "No Hallucination"
        else:
            # Find most common hallucination type among positive votes
            halluc_preds = predictions[predictions > 0]
            if len(halluc_preds) > 0:
                type_map = {
                    1: "Factual Hallucination",
                    2: "Contextual Hallucination", 
                    3: "Logical Hallucination",
                    4: "Multimodal Hallucination"
                }
                most_common_type = np.bincount(halluc_preds).argmax()
                predicted_type = type_map.get(most_common_type, "Factual Hallucination")
            else:
                predicted_type = "No Hallucination"
        
        result = PredictionResult(
            id=str(uuid.uuid4()),
            predicted_type=predicted_type,
            severity_percent=severity_percent
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.post("/batch-predict/{dataset_id}")
async def batch_predict(dataset_id: str):
    """Predict hallucinations for an entire dataset"""
    global trained_models, feature_vectorizer
    
    if not trained_models or feature_vectorizer is None:
        raise HTTPException(status_code=400, detail="No trained models available. Please train models first.")
    
    if dataset_id not in processed_data:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = processed_data[dataset_id].copy()
        
        # Extract features
        X, _ = create_feature_matrix(df, feature_vectorizer, fit=False)
        
        results = []
        
        for i in range(len(df)):
            X_single = X[i:i+1]
            
            # Get predictions from all models
            predictions = []
            for model in trained_models:
                pred = model.predict(X_single, verbose=0)
                pred_class = np.argmax(pred, axis=1)[0]
                predictions.append(pred_class)
            
            # Ensemble logic
            predictions = np.array(predictions)
            hallucination_votes = np.sum(predictions > 0)
            severity_percent = int((hallucination_votes / len(trained_models)) * 100)
            
            if hallucination_votes == 0:
                predicted_type = "No Hallucination"
            else:
                halluc_preds = predictions[predictions > 0]
                if len(halluc_preds) > 0:
                    type_map = {
                        1: "Factual Hallucination",
                        2: "Contextual Hallucination",
                        3: "Logical Hallucination", 
                        4: "Multimodal Hallucination"
                    }
                    most_common_type = np.bincount(halluc_preds).argmax()
                    predicted_type = type_map.get(most_common_type, "Factual Hallucination")
                else:
                    predicted_type = "No Hallucination"
            
            # Use existing ID if available, otherwise create new
            row_id = df.iloc[i].get('id', i)
            
            results.append({
                'id': row_id,
                'predicted_type': predicted_type,
                'severity_percent': severity_percent
            })
        
        # Save results to file
        results_df = pd.DataFrame(results)
        output_path = f"/tmp/{dataset_id}_predictions.csv"
        results_df.to_csv(output_path, index=False)
        
        return {
            "message": "Batch prediction completed",
            "total_predictions": len(results),
            "results_preview": results[:10],
            "download_path": f"/api/download-results/{dataset_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@api_router.get("/download-results/{dataset_id}")
async def download_results(dataset_id: str):
    """Download prediction results CSV"""
    file_path = f"/tmp/{dataset_id}_predictions.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        path=file_path,
        filename=f"{dataset_id}_hallucination_predictions.csv",
        media_type='text/csv'
    )

@api_router.get("/model-stats")
async def get_model_stats():
    """Get statistics about trained models"""
    global trained_models, training_status
    
    return {
        "models_trained": len(trained_models),
        "training_status": training_status,
        "model_ready": len(trained_models) > 0
    }

# Include the router in the main app

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")

@app.get("/")
def serve_react_app():
    return FileResponse("../frontend/build/index.html")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/datasets")
def get_datasets():
    return DATASETS

# @app.get("/api/datasets")
# def get_datasets():
#     # Example: return a list of datasets
#     return [
#         {
#             "id": "1",
#             "filename": "train.csv",
#             "rows": 100,
#             "columns": ["hyp", "tgt", "src", "label"]
#         }
#     ]


# @app.post("/api/train-models")
# def train_models():
#     return {"status": "training", "progress": 0, "message": "Training started..."}

# @app.post("/api/predict")
# def predict(data: dict):
#     return {"predicted_type": "A", "severity_percent": 80}

# @app.post("/api/batch-predict/{dataset_id}")
# def batch_predict(dataset_id: str):
#     return {"total_predictions": 100}

# @app.get("/api/download-results/{dataset_id}")
# def download_results(dataset_id: str):
#     # Implement file download logic here
#     return {"message": "Download started"}

@app.post("/api/eda/{dataset_id}")
def eda(dataset_id: str):
    # Find the file in your uploads directory
    file_path = os.path.join("uploads", dataset_id)
    if not os.path.exists(file_path):
        return {"error": "File not found"}, 404

    df = pd.read_csv(file_path)

    # Example: adjust these to your real column names
    label_col = "task"
    task_col = "task"
    hyp_col = "hyp"
    tgt_col = "tgt"

    # Class distribution
    class_dist = df[label_col].value_counts().to_dict()

    # Task distribution
    type_dist = df[task_col].value_counts(normalize=True).mul(100).round(1).to_dict()

    # Length stats
    hyp_avg = df[hyp_col].astype(str).apply(len).mean()
    tgt_avg = df[tgt_col].astype(str).apply(len).mean()
    length_stats = {"hyp_avg_length": int(hyp_avg), "tgt_avg_length": int(tgt_avg)}

    # Bar chart
    fig1, ax1 = plt.subplots()
    class_dist_keys = list(class_dist.keys())
    class_dist_vals = list(class_dist.values())
    ax1.bar(class_dist_keys, class_dist_vals)
    ax1.set_title("Distribution of Hallucination Labels")
    img1 = plot_to_base64(fig1)

    # Pie chart (img2)
    fig2, ax2 = plt.subplots()
    ax2.pie([37.5, 37.5, 25.1], labels=["DM", "MT", "PG"], autopct='%1.1f%%')
    ax2.set_title("Task Distribution")
    img2 = plot_to_base64(fig2)

    # Pie chart
    fig2, ax2 = plt.subplots()
    ax2.pie(type_dist.values(), labels=type_dist.keys(), autopct='%1.1f%%')
    ax2.set_title("Task Distribution")
    img2 = plot_to_base64(fig2)

    # Histogram
    hyp_lengths = df[hyp_col].astype(str).apply(len)
    tgt_lengths = df[tgt_col].astype(str).apply(len)
    fig3, ax3 = plt.subplots()
    ax3.hist(hyp_lengths, bins=30, alpha=0.7, label='Hyp Length')
    ax3.hist(tgt_lengths, bins=30, alpha=0.7, label='Tgt Length')
    ax3.set_title("Text Length Distribution")
    ax3.set_xlabel("Length")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    img3 = plot_to_base64(fig3)

    # Scatter plot
    fig4, ax4 = plt.subplots()
    ax4.scatter(hyp_lengths, tgt_lengths, alpha=0.5)
    ax4.set_title("Hypothesis vs Target Length")
    ax4.set_xlabel("Hyp Length")
    ax4.set_ylabel("Tgt Length")
    img4 = plot_to_base64(fig4)

    return {
        "class_distribution": class_dist,
        "type_distribution": type_dist,
        "length_stats": length_stats,
        "visualizations": [img1, img2, img3, img4]
    }

# @app.get("/api/training-status")
# def get_training_status():
#     return {"status": "completed", "progress": 100, "message": "Training done", "current_accuracy": 0.95}

@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Parse CSV for columns and row count
    with open(file_location, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        columns = next(reader)
        rows = sum(1 for _ in reader)
    dataset_info = {
        "id": file.filename,
        "filename": file.filename,
        "rows": rows,
        "columns": columns
    }
    DATASETS.append(dataset_info)
    return dataset_info

@app.get("/{full_path:path}")
def catch_all(full_path: str):
    file_path = f"../frontend/build/{full_path}"
    if os.path.exists(file_path) and not os.path.isdir(file_path):
        return FileResponse(file_path)
    return FileResponse("../frontend/build/index.html")

app.include_router(api_router)