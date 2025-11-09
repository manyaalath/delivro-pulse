# Revert Backend to Stable RandomForest State

## Tasks
- [ ] Update requirements.txt to keep only allowed dependencies
- [ ] Edit logistics_backend.py to use RandomForestClassifier, remove heavy deps, simplify code
- [ ] Test project runs with uvicorn logistics_backend:app --reload
- [ ] Confirm all endpoints (/train, /predict, /insights, /geo, /) respond correctly
- [ ] Clean up unused imports and functions
- [ ] Restore clean, stable structure
