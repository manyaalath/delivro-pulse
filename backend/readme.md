# 📦 Logistics Post-Mortem Analyzer - Backend API

FastAPI backend for analyzing and predicting delivery performance using machine learning.

## 🚀 Features

- **Machine Learning**: Train RandomForest models to predict late deliveries
- **Real-time Predictions**: API endpoint for delivery delay probability
- **Analytics Dashboard**: Comprehensive insights and metrics
- **AI-Powered Reports**: Generate executive summaries using Claude AI
- **Supabase Integration**: Seamless data fetching from your database

## 📋 Prerequisites

- Python 3.9+
- Supabase account with `deliveries` table
- Anthropic API key (for report generation)

## 🛠️ Installation

### 1. Clone and Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### 3. Verify Supabase Table

Ensure your Supabase `deliveries` table has these columns:
- `package_id` (text)
- `courier_id` (text)
- `city` (text)
- `region_id` (text)
- `lat` (float8)
- `lng` (float8)
- `accept_time` (timestamp)
- `delivery_time` (timestamp)
- `delay_min` (int8)
- `status` (text)

## 🏃 Running the Server

### Local Development

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: `http://localhost:8000`

### API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📡 API Endpoints

### 1. **POST /train**
Train the prediction model with latest data from Supabase.

**Response:**
```json
{
  "status": "trained",
  "accuracy": 0.8745,
  "f1_score": 0.8423,
  "precision": 0.8654,
  "recall": 0.8201,
  "message": "Model trained successfully on 5000 records"
}
```

### 2. **GET /predict**
Predict delivery delay probability.

**Query Parameters:**
- `city` (string): City name
- `region_id` (string): Region ID
- `courier_id` (string): Courier ID
- `hour` (int): Hour of delivery (0-23)
- `weekday` (int): Weekday (0=Monday, 6=Sunday)
- `lat` (float): Latitude
- `lng` (float): Longitude
- `duration_hours` (float): Expected duration

**Example:**
```bash
GET /predict?city=Delhi&region_id=N1&courier_id=C204&hour=18&weekday=4&lat=28.6139&lng=77.2090&duration_hours=2.5
```

**Response:**
```json
{
  "late_probability": 0.7342,
  "prediction": "likely_late",
  "confidence": 0.7342
}
```

### 3. **GET /insights**
Get comprehensive analytics insights.

**Response:**
```json
{
  "most_delayed_city": "Delhi",
  "best_courier": "C204",
  "avg_delay_by_region": {
    "N1": 8.4,
    "S2": 5.6,
    "W3": 10.2
  },
  "model_metrics": {
    "accuracy": 0.8745,
    "precision": 0.8654,
    "recall": 0.8201,
    "f1_score": 0.8423
  },
  "total_deliveries": 5000,
  "late_delivery_rate": 0.3245
}
```

### 4. **GET /generate_report**
Generate an AI-powered executive summary.

**Response:**
```json
{
  "report": "The logistics network processed 5,000 deliveries with a 32.4% late delivery rate...",
  "generated_at": "2025-11-09T10:30:00"
}
```

### 5. **GET /health**
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-09T10:30:00"
}
```

## 🔄 Typical Workflow

1. **Upload Data**: Lovable frontend uploads CSV → Supabase `deliveries` table
2. **Train Model**: Call `POST /train` to train with latest data
3. **Get Insights**: Frontend calls `GET /insights` for dashboard
4. **Predictions**: Use `GET /predict` for simulation scenarios
5. **Generate Report**: Call `GET /generate_report` for AI summary

## 🧪 Testing with cURL

```bash
# Train the model
curl -X POST http://localhost:8000/train

# Get insights
curl http://localhost:8000/insights

# Make a prediction
curl "http://localhost:8000/predict?city=Delhi&region_id=N1&courier_id=C204&hour=18&weekday=4&lat=28.6139&lng=77.2090&duration_hours=2.5"

# Generate report
curl http://localhost:8000/generate_report
```

## 🔒 CORS Configuration

Update CORS settings in `main.py` for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-lovable-frontend.lovable.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 Model Details

- **Algorithm**: RandomForestClassifier
- **Features**: City, Region, Courier, Hour, Weekday, Coordinates, Duration
- **Target**: Binary classification (late vs on-time)
- **Evaluation**: Accuracy, Precision, Recall, F1-Score
- **Storage**: Model saved as `model.pkl` for persistence

## 🐛 Troubleshooting

### Model Not Found Error
```bash
# Train the model first
curl -X POST http://localhost:8000/train
```

### Supabase Connection Error
- Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- Check table name is exactly `deliveries`
- Ensure Supabase service is running

### No Data Error
- Upload CSV data through Lovable frontend
- Verify data exists in Supabase table

## 📦 Deployment Options

### Replit
1. Import GitHub repository
2. Add environment variables in Secrets
3. Run automatically

### Railway
```bash
railway login
railway init
railway up
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Performance Tips

1. **Cache Model**: Model loads once and stays in memory
2. **Batch Predictions**: Process multiple predictions efficiently
3. **Database Indexing**: Index `city`, `region_id`, `courier_id` in Supabase
4. **Rate Limiting**: Add rate limiting for production use

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - feel free to use for your projects.

---

**Built with ❤️ using FastAPI, Scikit-learn, and Claude AI**