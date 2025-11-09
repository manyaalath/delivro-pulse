# TODO: Implement Generate Insights Button Functionality

## Tasks
- [x] Modify `src/components/InsightsButton.tsx`:
  - [x] Change `generateInsights` function to POST to `http://localhost:8000/train` (optional retrain), then GET `http://localhost:8000/insights`
  - [x] Update state to hold model metrics (accuracy, precision, recall, f1) instead of insights text
  - [x] Replace dialog content to display metrics in a modal
  - [x] Ensure loading state and error handling with toasts
  - [x] Add success log "🧠 Insights refreshed"
  - [x] Optionally trigger refresh of ModelPerformance component
- [x] Update `src/components/ModelPerformance.tsx` to support ref-based refresh
- [x] Update `src/pages/Index.tsx` to pass refresh callback to InsightsButton

## Followup Steps
- [ ] Test the button with backend running on localhost:8000
- [ ] Verify modal displays metrics correctly
- [ ] Ensure error toasts and loading states work
