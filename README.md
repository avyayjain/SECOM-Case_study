# SECOM Semiconductor Fault Detection

An educational case study of the SECOM semiconductor manufacturing dataset. The project explores data quality, preprocessing, feature selection, model comparison, class imbalance, decision-threshold tuning, and evaluation for identifying potentially faulty production observations. It also includes a FastAPI backend and Streamlit interface for exploring the workflow and scoring sensor readings.

> Educational prototype only. Predictions and model scores are not validated for production use. A flagged observation is a candidate for inspection, not proof of a defect.

## Dataset

The files in `data/` contain the SECOM sensor measurements and labels. The sensor file has 1,567 observations and 590 sensor features; `NaN` values represent missing readings. The label file uses `-1` for pass and `+1` for fault. The project maps these to `0` and `1`, respectively.

The dataset originates from the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom). Keep the sensor and label rows in their original order so observations and labels remain aligned.

## Setup

Use Python with the packages listed in `requirements.txt`. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can invoke the environment's Python directly as `.venv\Scripts\python.exe`.

## Run the application

The repository includes a trained model and its decision configuration in `results/final_model/`. To regenerate them from the current dataset, run `python final_inspection_workflow.py` from the project root.

Start the API and interface in separate terminals, from the project root:

```powershell
python -m uvicorn app.api:app --reload
```

```powershell
python -m streamlit run app/frontend.py
```

Open the Streamlit URL printed in the terminal (typically `http://localhost:8501`). The API runs at `http://127.0.0.1:8000`; its interactive documentation is available at `http://127.0.0.1:8000/docs`.

## Analysis scripts

The standalone Python scripts run from the project root and cover the analysis workflow:

- Exploration and baselines: `eda.py`, `baseline.py`
- Feature selection and importance: `feature_selection.py`, `forest_feature_selection.py`, `permutation_importance.py`, `feature_stability.py`, `feature_importance_error_analysis.py`
- Model evaluation and comparison: `compare_models.py`, `compare_xgboost.py`, `cross_validation.py`, `nested_cv.py`, `rigorous_model_comparison.py`, `validate_models.py`
- Threshold, cost, and calibration: `threshold_tuning.py`, `cost_sensitive_eval.py`, `cost_sensitive_evaluation.py`, `probability_calibration.py`, `tune_selected_models.py`
- Final workflow: `final_inspection_workflow.py`

Some experiments may require a compatible XGBoost or GPU setup; see `requirements.txt` for the project's declared dependencies. Scripts may write generated tables, figures, or model artifacts under `results/`.

## Project layout

```text
app/                  FastAPI backend and Streamlit frontend
data/                 SECOM sensor readings and labels
results/              Generated figures, tables, and saved model
*.py                  Standalone analysis and modeling workflows
requirements.txt      Python dependencies
```
