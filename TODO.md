# DELIVRO-PULSE Backend Setup and Testing TODO

## 1. Edit backend/main.py
- [x] Add train_lade_model() function to load LaDe dataset, preprocess, train model, save model and metrics.
- [x] Modify /train endpoint to call train_lade_model() instead of current Supabase-based training.
- [x] Append Supabase connection check code after training function.

## 2. Update backend/requirements.txt
- [x] Ensure all required dependencies are listed: datasets, supabase-python, python-dotenv, pandas, scikit-learn, joblib, fastapi, uvicorn.

## 3. Backend Setup
- [x] Navigate to backend directory.
- [x] Create virtual environment: python -m venv venv.
- [x] Activate venv: venv\Scripts\activate (Windows).
- [x] Install dependencies: pip install -r requirements.txt.

## 4. Run Backend Server
- [x] Start server: uvicorn main:app --reload.

## 5. Test API Routes
- [x] Test /train: curl -X POST http://localhost:8000/train
- [x] Test /predict: curl "http://localhost:8000/predict?city=1&region_id=2&courier_id=3&hour=14&weekday=3"
- [x] Test /insights: curl http://localhost:8000/insights

## 6. Frontend Integration Test
- [x] Run frontend: npm run dev (from root).
- [x] Open Lovable dashboard, click "🧠 Generate Insights" button.
- [x] Verify it fetches from http://localhost:8000/insights and shows JSON with metrics.

## 7. Final Verification Summary
- [x] Print final summary of connection status.
