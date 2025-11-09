from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import anthropic
import json
from datasets import load_dataset

# Initialize FastAPI
app = FastAPI(title="📦 Logistics Post-Mortem Analyzer API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Lovable frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "your-supabase-url")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-supabase-key")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-key")

# Initialize clients
supabase: Client = None
if SUPABASE_URL != "your-supabase-url" and SUPABASE_KEY != "your-supabase-key":
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Failed to initialize Supabase client: {e}")
        supabase = None

# Global variables for model and encoders
model = None
encoders = {}
feature_columns = []
model_metrics = {}

# Pydantic Models
class TrainResponse(BaseModel):
    status: str
    model: str
    metrics: Dict[str, float]
    message: str

class PredictResponse(BaseModel):
    late_probability: float
    prediction: str
    confidence: float

class InsightsResponse(BaseModel):
    most_delayed_city: str
    best_courier: str
    avg_delay_by_region: Dict[str, float]
    model_metrics: Dict[str, float]
    total_deliveries: int
    late_delivery_rate: float

class ReportResponse(BaseModel):
    report: str
    generated_at: str

# Helper Functions
def train_lade_model():
    print("🚀 Loading LaDe dataset...")
    dataset = load_dataset("Cainiao-AI/LaDe", "LaDe-D")
    df = pd.DataFrame(dataset['train'])

    # Select only relevant columns
    cols = ['package_id','courier_id','city','region_id','lng','lat','accept_time','delivery_time']
    df = df[[c for c in cols if c in df.columns]].dropna()

    print("🧹 Preprocessing data...")
    df['accept_time'] = pd.to_datetime(df['accept_time'])
    df['delivery_time'] = pd.to_datetime(df['delivery_time'])
    df['delay_min'] = (df['delivery_time'] - df['accept_time']).dt.total_seconds() / 60
    df['late_flag'] = (df['delay_min'] > 0).astype(int)

    # Feature engineering
    df['hour'] = df['delivery_time'].dt.hour
    df['weekday'] = df['delivery_time'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
    df['delivery_duration'] = df['delay_min'].clip(lower=0, upper=300)  # clip outliers

    # Encode categoricals
    encoders = {}
    for col in ['city','region_id','courier_id']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Features and label
    features = ['city','region_id','courier_id','hour','weekday','is_weekend','delivery_duration']
    X = df[features]
    y = df['late_flag']

    # Downsample for faster testing if huge
    if len(X) > 200000:
        df = df.sample(200000, random_state=42)
        X = df[features]
        y = df['late_flag']

    # Balance the dataset
    print("⚖ Balancing classes with SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    # ---- MODEL 1: LIGHTGBM ----
    print("🌲 Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        class_weight='balanced'
    )
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)

    # ---- MODEL 2: CATBOOST ----
    print("🐈 Training CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=400,
        depth=8,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='F1',
        verbose=False,
        auto_class_weights='Balanced'
    )
    cat_model.fit(X_train, y_train)
    cat_preds = cat_model.predict(X_test)

    # ---- EVALUATE ----
    def metrics(y_true, y_pred):
        return {
            "accuracy": round(accuracy_score(y_true, y_pred)*100, 2),
            "precision": round(precision_score(y_true, y_pred)*100, 2),
            "recall": round(recall_score(y_true, y_pred)*100, 2),
            "f1_score": round(f1_score(y_true, y_pred)*100, 2)
        }

    lgb_metrics = metrics(y_test, lgb_preds)
    cat_metrics = metrics(y_test, cat_preds)

    print("LightGBM:", lgb_metrics)
    print("CatBoost:", cat_metrics)

    # Choose best model
    if cat_metrics['f1'] >= lgb_metrics['f1']:
        best_model, best_metrics, best_name = cat_model, cat_metrics, "CatBoost"
    else:
        best_model, best_metrics, best_name = lgb_model, lgb_metrics, "LightGBM"

    print(f"🏆 Best Model: {best_name}")
    print(f"✅ Accuracy: {best_metrics['accuracy']}% | F1: {best_metrics['f1']}%")

    # Save model and metrics
    joblib.dump(best_model, "model_best.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(features, "feature_columns.pkl")
    with open("metrics.json", "w") as f: json.dump(best_metrics, f)
    print("📦 Model and metrics saved.")

    return best_name, best_metrics

# Supabase connection check
from dotenv import load_dotenv
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Test query
try:
    data = supabase.table("deliveries").select("*").limit(1).execute()
    print("✅ Supabase connection OK")
except Exception as e:
    print("❌ Supabase connection failed:", e)

def fetch_data_from_supabase():
    """Fetch all delivery data from Supabase or use synthetic data if connection fails"""
    try:
        if supabase is None:
            raise Exception("Supabase not connected")

        response = supabase.table('deliveries').select('*').execute()
        df = pd.DataFrame(response.data)

        if df.empty:
            raise ValueError("No data found in deliveries table")

        return df
    except Exception as e:
        print(f"Supabase connection failed: {e}. Using synthetic data instead.")
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1000

        synthetic_data = {
            'package_id': [f'PKG_{i}' for i in range(n_samples)],
            'courier_id': np.random.randint(1, 20, n_samples),
            'city': np.random.randint(1, 10, n_samples),
            'region_id': np.random.randint(1, 5, n_samples),
            'lat': np.random.uniform(30, 40, n_samples),
            'lng': np.random.uniform(100, 120, n_samples),
            'accept_time': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
            'delivery_time': pd.date_range('2024-01-01 01:00:00', periods=n_samples, freq='1H'),
            'delay_min': np.random.exponential(5, n_samples)  # Exponential delay
        }

        df = pd.DataFrame(synthetic_data)
        return df

def preprocess_data(df: pd.DataFrame):
    """Clean and preprocess the data"""
    # Create a copy
    df = df.copy()
    
    # Drop rows with null delay_min
    df = df.dropna(subset=['delay_min'])
    
    # Ensure delay_min is numeric
    df['delay_min'] = pd.to_numeric(df['delay_min'], errors='coerce')
    df = df.dropna(subset=['delay_min'])
    
    # Create target label
    df['late_flag'] = (df['delay_min'] > 0).astype(int)
    
    # Convert timestamps
    df['delivery_time'] = pd.to_datetime(df['delivery_time'])
    df['accept_time'] = pd.to_datetime(df['accept_time'])
    
    # Extract time features
    df['hour'] = df['delivery_time'].dt.hour
    df['weekday'] = df['delivery_time'].dt.weekday
    df['day'] = df['delivery_time'].dt.day
    df['month'] = df['delivery_time'].dt.month
    
    # Calculate delivery duration
    df['duration_hours'] = (df['delivery_time'] - df['accept_time']).dt.total_seconds() / 3600
    
    return df

def encode_features(df: pd.DataFrame, fit=True):
    """Encode categorical features"""
    global encoders
    
    categorical_cols = ['city', 'region_id', 'courier_id']
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            if fit:
                encoders[col] = LabelEncoder()
                df_encoded[f'{col}_encoded'] = encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                df_encoded[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1
                )
    
    return df_encoded

def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix for model"""
    feature_cols = [
        'city_encoded', 'region_id_encoded', 'courier_id_encoded',
        'hour', 'weekday', 'day', 'month', 'duration_hours',
        'lat', 'lng'
    ]
    
    # Select only columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[available_cols]
    y = df['late_flag']
    
    return X, y, available_cols

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "📦 Logistics Post-Mortem Analyzer API",
        "version": "1.0.0",
        "endpoints": ["/train", "/predict", "/insights", "/generate_report"]
    }

@app.post("/train", response_model=TrainResponse)
async def train_model():
    """Train the delivery prediction model using LaDe dataset"""
    global model, feature_columns, model_metrics

    try:
        # Train model using LaDe dataset
        model_name, metrics = train_lade_model()

        # Store metrics
        model_metrics = metrics

        return TrainResponse(
            status="trained",
            model=model_name,
            metrics=metrics,
            message=f"Model trained successfully using {model_name}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/predict", response_model=PredictResponse)
async def predict(
    city: str = Query(..., description="City name"),
    region_id: str = Query(..., description="Region ID"),
    courier_id: str = Query(..., description="Courier ID"),
    hour: int = Query(..., ge=0, le=23, description="Hour of delivery"),
    weekday: int = Query(..., ge=0, le=6, description="Weekday (0=Monday)"),
    lat: float = Query(default=0.0, description="Latitude"),
    lng: float = Query(default=0.0, description="Longitude"),
    duration_hours: float = Query(default=2.0, description="Duration in hours")
):
    """Predict delivery delay probability"""
    global model, encoders, feature_columns

    try:
        # Load model if not in memory
        if model is None:
            if not os.path.exists("model_best.pkl"):
                raise HTTPException(status_code=400, detail="Model not trained. Please call /train first.")
            model = joblib.load("model_best.pkl")
            encoders = joblib.load("encoders.pkl")
            feature_columns = joblib.load("feature_columns.pkl")

        # Prepare input data
        input_data = {
            'city': city,
            'region_id': region_id,
            'courier_id': courier_id,
            'hour': hour,
            'weekday': weekday,
            'is_weekend': 1 if weekday in [5, 6] else 0,
            'delivery_duration': duration_hours * 60  # Convert to minutes
        }

        df_input = pd.DataFrame([input_data])

        # Encode categorical features
        for col in ['city', 'region_id', 'courier_id']:
            if col in encoders:
                try:
                    df_input[col] = encoders[col].transform([str(df_input[col].iloc[0])])
                except:
                    df_input[col] = -1  # Unknown category

        # Select features
        X_pred = df_input[feature_columns]

        # Predict
        probability = model.predict_proba(X_pred)[0][1]  # Probability of being late
        prediction = "likely_late" if probability > 0.5 else "likely_on_time"
        confidence = probability if probability > 0.5 else (1 - probability)

        return PredictResponse(
            late_probability=round(probability, 4),
            prediction=prediction,
            confidence=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/insights", response_model=InsightsResponse)
async def get_insights():
    """Get analytics insights from the data"""
    try:
        # Fetch data
        df = fetch_data_from_supabase()
        df = preprocess_data(df)
        
        # Most delayed city
        city_delays = df.groupby('city')['delay_min'].mean().sort_values(ascending=False)
        most_delayed_city = str(city_delays.index[0]) if len(city_delays) > 0 else "N/A"

        # Best courier (minimum 20 deliveries)
        courier_stats = df.groupby('courier_id').agg({
            'delay_min': 'mean',
            'package_id': 'count'
        }).rename(columns={'package_id': 'count'})

        eligible_couriers = courier_stats[courier_stats['count'] >= 20]
        best_courier = str(eligible_couriers['delay_min'].idxmin()) if len(eligible_couriers) > 0 else "N/A"

        # Average delay by region
        region_delays = df.groupby('region_id')['delay_min'].mean().round(2).to_dict()
        region_delays = {str(k): v for k, v in region_delays.items()}
        
        # Late delivery rate
        total_deliveries = len(df)
        late_deliveries = df['late_flag'].sum()
        late_rate = (late_deliveries / total_deliveries) if total_deliveries > 0 else 0
        
        # Load model metrics
        global model_metrics
        if not model_metrics:
            if os.path.exists("metrics.json"):
                with open("metrics.json", "r") as f:
                    model_metrics = json.load(f)
                # Normalize keys to match frontend expectations
                model_metrics = {
                    "accuracy": model_metrics.get("accuracy", 0) / 100,  # Convert to decimal
                    "f1_score": model_metrics.get("f1", 0) / 100,
                    "precision": model_metrics.get("precision", 0) / 100,
                    "recall": model_metrics.get("recall", 0) / 100
                }
            else:
                model_metrics = {
                    "accuracy": 0.0,
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0
                }
        
        return InsightsResponse(
            most_delayed_city=most_delayed_city,
            best_courier=str(best_courier),
            avg_delay_by_region=region_delays,
            model_metrics=model_metrics,
            total_deliveries=total_deliveries,
            late_delivery_rate=round(late_rate, 4)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")

@app.get("/generate_report", response_model=ReportResponse)
async def generate_report():
    """Generate a human-readable report using Claude AI"""
    try:
        # Get insights
        insights = await get_insights()
        
        # Prepare prompt for Claude
        prompt = f"""Generate a concise executive summary for a logistics delivery analysis with the following data:

- Total Deliveries: {insights.total_deliveries}
- Late Delivery Rate: {insights.late_delivery_rate * 100:.1f}%
- Most Delayed City: {insights.most_delayed_city}
- Best Performing Courier: {insights.best_courier}
- Regional Performance: {json.dumps(insights.avg_delay_by_region, indent=2)}
- Model Performance: Accuracy {insights.model_metrics.get('accuracy', 0)*100:.1f}%, F1-Score {insights.model_metrics.get('f1_score', 0):.2f}

Write a 3-4 paragraph professional report that:
1. Summarizes overall delivery performance
2. Highlights key problem areas and top performers
3. Provides actionable recommendations
Keep it concise and business-focused."""

        # Call Claude API
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        report_text = message.content[0].text
        
        return ReportResponse(
            report=report_text,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/geo")
async def get_geo_data():
    """Get geospatial data for map visualization"""
    try:
        # Fetch data
        df = fetch_data_from_supabase()
        df = preprocess_data(df)

        # Group by city/region and calculate average delay
        geo_data = df.groupby(['city', 'region_id']).agg({
            'lat': 'mean',
            'lng': 'mean',
            'delay_min': 'mean'
        }).reset_index()

        # Convert to expected format
        points = []
        for _, row in geo_data.iterrows():
            points.append({
                'city': str(row['city']),
                'region_id': str(row['region_id']),
                'lat': float(row['lat']),
                'lng': float(row['lng']),
                'avg_delay': round(float(row['delay_min']), 2)
            })

        return {"points": points}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geo data generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
