import logging
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="📦 DELIVRO-PULSE Logistics Backend API")

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

# Initialize Supabase client
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

class GeoResponse(BaseModel):
    points: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Helper Functions
def train_lade_model():
    print("🚀 Loading LaDe dataset...")
    df = pd.read_csv("https://huggingface.co/datasets/Cainiao-AI/LaDe/resolve/main/delivery/delivery_bj.csv")

    # Select relevant columns
    cols = ['package_id', 'courier_id', 'city', 'region_id', 'lng', 'lat', 'accept_time', 'delivery_time']
    df = df[[c for c in cols if c in df.columns]].dropna()

    print("🧹 Preprocessing data...")
    df['accept_time'] = pd.to_datetime(df['accept_time'])
    df['delivery_time'] = pd.to_datetime(df['delivery_time'])
    df['delay_min'] = (df['delivery_time'] - df['accept_time']).dt.total_seconds() / 60
    df['late_flag'] = (df['delay_min'] > 0).astype(int)

    # Feature engineering
    df['hour'] = df['delivery_time'].dt.hour
    df['weekday'] = df['delivery_time'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['delivery_duration'] = df['delay_min'].clip(lower=0, upper=300)

    # Encode categoricals
    encoders = {}
    for col in ['city', 'region_id', 'courier_id']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Features and label
    features = ['city', 'region_id', 'courier_id', 'hour', 'weekday', 'is_weekend', 'delivery_duration']
    X = df[features]
    y = df['late_flag']

    # Downsample if too large
    if len(X) > 200000:
        df = df.sample(200000, random_state=42)
        X = df[features]
        y = df['late_flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train RandomForest with balanced class weights
    print("🌲 Training RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, rf_preds) * 100, 2),
        "precision": round(precision_score(y_test, rf_preds) * 100, 2),
        "recall": round(recall_score(y_test, rf_preds) * 100, 2),
        "f1": round(f1_score(y_test, rf_preds) * 100, 2)
    }

    print("RandomForest Metrics:", metrics)

    # Save model and artifacts
    joblib.dump(rf_model, "model.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(features, "feature_columns.pkl")
    with open("metrics.json", "w") as f:
        import json
        json.dump(metrics, f)
    print("📦 Model and metrics saved.")

    return "RandomForest", metrics

def fetch_data_from_supabase():
    """Fetch delivery data from Supabase or use synthetic data"""
    try:
        if supabase is None:
            raise Exception("Supabase not connected")

        response = supabase.table('deliveries').select('*').execute()
        df = pd.DataFrame(response.data)

        if df.empty:
            raise ValueError("No data found in deliveries table")

        return df
    except Exception as e:
        print(f"Supabase connection failed: {e}. Using synthetic data.")
        # Synthetic data
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
            'delay_min': np.random.exponential(5, n_samples)
        }

        df = pd.DataFrame(synthetic_data)
        return df

def preprocess_data(df: pd.DataFrame):
    """Clean and preprocess data"""
    df = df.copy()
    df = df.dropna(subset=['delay_min'])
    df['delay_min'] = pd.to_numeric(df['delay_min'], errors='coerce')
    df = df.dropna(subset=['delay_min'])
    df['late_flag'] = (df['delay_min'] > 0).astype(int)
    df['delivery_time'] = pd.to_datetime(df['delivery_time'])
    df['accept_time'] = pd.to_datetime(df['accept_time'])
    df['hour'] = df['delivery_time'].dt.hour
    df['weekday'] = df['delivery_time'].dt.weekday
    df['day'] = df['delivery_time'].dt.day
    df['month'] = df['delivery_time'].dt.month
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
                df_encoded[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1
                )
    return df_encoded

def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix"""
    feature_cols = [
        'city_encoded', 'region_id_encoded', 'courier_id_encoded',
        'hour', 'weekday', 'day', 'month', 'duration_hours',
        'lat', 'lng'
    ]
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols]
    y = df['late_flag']
    return X, y, available_cols

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "📦 DELIVRO-PULSE Logistics Backend API",
        "version": "1.0.0",
        "endpoints": ["/welcome", "/train", "/predict", "/insights", "/geo", "/health"]
    }

@app.get("/welcome")
async def welcome(request: Request):
    """Returns a welcome message and logs request metadata"""
    logger.info(f"Request received: {request.method} {request.url.path}")
    return {"message": "Welcome to the DELIVRO-PULSE Logistics Backend API!"}

@app.post("/train", response_model=TrainResponse)
async def train_model():
    """Train the RandomForest model using LaDe dataset"""
    global model, feature_columns, model_metrics
    try:
        model_name, metrics = train_lade_model()
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
        if model is None:
            if not os.path.exists("model.pkl"):
                raise HTTPException(status_code=400, detail="Model not trained. Please call /train first.")
            model = joblib.load("model.pkl")
            encoders = joblib.load("encoders.pkl")
            feature_columns = joblib.load("feature_columns.pkl")

        input_data = {
            'city': city,
            'region_id': region_id,
            'courier_id': courier_id,
            'hour': hour,
            'weekday': weekday,
            'is_weekend': 1 if weekday in [5, 6] else 0,
            'delivery_duration': duration_hours * 60
        }
        df_input = pd.DataFrame([input_data])
        for col in ['city', 'region_id', 'courier_id']:
            if col in encoders:
                try:
                    df_input[col] = encoders[col].transform([str(df_input[col].iloc[0])])
                except:
                    df_input[col] = -1
        X_pred = df_input[feature_columns]
        probability = model.predict_proba(X_pred)[0][1]
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
    """Get analytics insights"""
    try:
        df = fetch_data_from_supabase()
        df = preprocess_data(df)
        city_delays = df.groupby('city')['delay_min'].mean().sort_values(ascending=False)
        most_delayed_city = str(city_delays.index[0]) if len(city_delays) > 0 else "N/A"
        courier_stats = df.groupby('courier_id').agg({
            'delay_min': 'mean',
            'package_id': 'count'
        }).rename(columns={'package_id': 'count'})
        eligible_couriers = courier_stats[courier_stats['count'] >= 20]
        best_courier = str(eligible_couriers['delay_min'].idxmin()) if len(eligible_couriers) > 0 else "N/A"
        region_delays = df.groupby('region_id')['delay_min'].mean().round(2).to_dict()
        region_delays = {str(k): v for k, v in region_delays.items()}
        total_deliveries = len(df)
        late_deliveries = df['late_flag'].sum()
        late_rate = (late_deliveries / total_deliveries) if total_deliveries > 0 else 0
        global model_metrics
        if not model_metrics and os.path.exists("metrics.json"):
            import json
            with open("metrics.json", "r") as f:
                model_metrics = json.load(f)
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

@app.get("/geo", response_model=GeoResponse)
async def get_geo_data():
    """Get geo data for Mapbox visualization"""
    try:
        base_url = "https://huggingface.co/datasets/Cainiao-AI/LaDe/raw/main/delivery/"
        cities = ["delivery_bj.csv", "delivery_sh.csv", "delivery_gz.csv"]
        dfs = []

        for file in cities:
            url = f"{base_url}{file}"
            df_city = pd.read_csv(url)
            dfs.append(df_city)

        df = pd.concat(dfs, ignore_index=True)

        # Keep safe columns only
        keep_cols = ["city","region_id","accept_time","delivery_time","accept_gps_lat","accept_gps_lng"]
        df = df[[c for c in keep_cols if c in df.columns]].dropna(subset=["accept_time","delivery_time"])

        # Normalize column names
        df.rename(columns={
            "accept_gps_lat": "lat",
            "accept_gps_lng": "lng"
        }, inplace=True)

        # Time parsing & delay calculation
        df["accept_time"] = pd.to_datetime(df["accept_time"], errors="coerce")
        df["delivery_time"] = pd.to_datetime(df["delivery_time"], errors="coerce")
        df["delay_min"] = (df["delivery_time"] - df["accept_time"]).dt.total_seconds() / 60
        df = df.dropna(subset=["lat","lng","delay_min"])

        # Group by city and compute average delay + mean coordinates
        grouped = (
            df.groupby(["city"])
            .agg({"lat":"mean","lng":"mean","delay_min":"mean"})
            .reset_index()
        )
        grouped.rename(columns={"delay_min": "avg_delay"}, inplace=True)

        points = grouped.to_dict(orient="records")

        print(f"✅ Geo data prepared successfully: {len(points)} city points.")
        return GeoResponse(points=points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geo data retrieval failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
