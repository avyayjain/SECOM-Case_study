import io
import math
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"

API_URL = "http://127.0.0.1:8000"
try:
    if "API_URL" in st.secrets:
        API_URL = st.secrets["API_URL"]
except Exception:
    pass

st.set_page_config(
    page_title="SECOM Case Study",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 SECOM Fault Detection — Case Study")
st.caption(
    "From raw semiconductor sensor data to a cost-sensitive machine-learning inspection decision"
)

st.warning(
    "Educational prototype & final evaluation dashboard. Model scores are not guaranteed "
    "real-world fault probabilities. Do not use these predictions to release or reject production "
    "material without independent validation."
)


# ---------------------------------------------------------
# API & Data helpers
# ---------------------------------------------------------

def get_api(path):
    response = requests.get(f"{API_URL}{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def post_api(path, payload):
    response = requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def load_local_csv(filename: str):
    path = TABLES_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def display_figure(figure_name: str, caption: str = ""):
    local_path = FIGURES_DIR / figure_name
    if local_path.exists():
        st.image(str(local_path), caption=caption, use_container_width=True)
    else:
        try:
            res = requests.get(f"{API_URL}/case-study/figures/{figure_name}", timeout=10)
            if res.status_code == 200:
                st.image(res.content, caption=caption, use_container_width=True)
        except Exception:
            pass


def show_table(data, key_column=None):
    """Display dictionaries or lists as a readable table."""
    if isinstance(data, list) and data:
        st.dataframe(
            pd.DataFrame(data),
            use_container_width=True,
            hide_index=True,
        )
    elif isinstance(data, dict):
        st.dataframe(
            pd.DataFrame(
                [{"Metric": key, "Value": value} for key, value in data.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No tabular results available.")


def show_lesson(text):
    st.info(f"**What this means:** {text}")


def run_stage(label, endpoint):
    """Fetch one case-study stage and retain the result for this session."""
    try:
        result = get_api(endpoint)
        st.session_state[f"stage:{endpoint}"] = result
        return result
    except requests.RequestException as exc:
        st.error(
            f"Could not load {label}. Check that FastAPI is running at "
            f"{API_URL}.\n\nDetails: {exc}"
        )
        return None


def get_stage(endpoint):
    return st.session_state.get(f"stage:{endpoint}")


# ---------------------------------------------------------
# Sidebar: API status and navigation
# ---------------------------------------------------------

with st.sidebar:
    st.header("System status")

    try:
        health = get_api("/health")
        if health.get("status") == "ok":
            st.success("FastAPI connected")
            st.caption(f"Configured Threshold: `{health.get('threshold', 0.495)}`")
        else:
            st.error("Backend reports an error")
            st.json(health)
    except Exception:
        st.error("FastAPI is not reachable")

    st.caption(f"Backend: {API_URL}")

    st.divider()
    st.subheader("Case-study stages")

    st.markdown(
        """
        1. Dataset overview
        2. Data quality and EDA
        3. Preprocessing & SIP Log
        4. Model experiments
        5. Threshold and cost
        6. Final model evaluation
        7. Predict new readings
        """
    )

    if st.button("Clear displayed results", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("stage:") or key.startswith("prediction:") or key.startswith("batch:"):
                del st.session_state[key]
        st.rerun()


# ---------------------------------------------------------
# Workflow progress
# ---------------------------------------------------------

stage_endpoints = [
    "/case-study/overview",
    "/case-study/data-quality",
    "/case-study/preprocessing",
    "/case-study/experiments",
    "/case-study/threshold",
    "/case-study/final-model",
]

completed = sum(
    1 for endpoint in stage_endpoints
    if get_stage(endpoint) is not None
)

st.progress(completed / len(stage_endpoints))
st.caption(f"Analysis stages loaded: {completed} of {len(stage_endpoints)}")


# =========================================================
# STEP 1 — Understand the dataset
# =========================================================

st.divider()
st.header("Step 1 — Understand the dataset")

st.markdown(
    """
    Before training a model, establish what the data contains and what the
    model is expected to detect.

    In SECOM, the original labels are **-1 = Pass** and **+1 = Fault**.
    The API converts them into **0 = Pass** and **1 = Fault** for analysis.
    """
)

if st.button("Run Step 1: Dataset overview", key="step1"):
    run_stage("dataset overview", "/case-study/overview")

overview = get_stage("/case-study/overview")

if overview:
    cols = st.columns(4)
    cols[0].metric("Observations", f"{overview['observations']:,}")
    cols[1].metric("Sensor features", overview["features"])
    cols[2].metric("Pass samples", f"{overview['pass_count']:,} ({overview['pass_percent']}%)")
    cols[3].metric("Fault samples", f"{overview['fault_count']} ({overview['fault_percent']}%)")

    st.subheader("Class distribution")

    distribution = pd.DataFrame(
        {
            "Class": ["Pass", "Fault"],
            "Samples": [overview["pass_count"], overview["fault_count"]],
            "Percentage": [overview["pass_percent"], overview["fault_percent"]],
        }
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(distribution, use_container_width=True, hide_index=True)
        st.caption(f"Severe Class Imbalance: **{overview.get('imbalance_ratio', '14.1:1')}**")

    with col2:
        display_figure("class_distribution.png", caption="Wafer Class Distribution: Pass vs Fault")

    show_lesson(overview["lesson"])


# =========================================================
# STEP 2 — Data quality and EDA
# =========================================================

st.divider()
st.header("Step 2 — Data quality and exploratory analysis")

st.markdown(
    """
    Check missing readings before choosing a preprocessing method. Missing
    values are not automatically faults, and missingness alone does not prove
    a sensor is defective.
    """
)

if st.button("Run Step 2: Data quality", key="step2"):
    run_stage("data quality analysis", "/case-study/data-quality")

quality = get_stage("/case-study/data-quality")

if quality:
    cols = st.columns(4)
    cols[0].metric("Missing cells", f'{quality["missing_cells"]:,}')
    cols[1].metric("Missing cell rate", f'{quality["missing_cell_percent"]:.2f}%')
    cols[2].metric("Features >60% missing", quality.get("features_gt_60_percent_missing", 24))
    cols[3].metric("Duplicate rows", quality["duplicate_rows"])

    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        st.subheader("Sensor Missingness Distribution")
        display_figure("missingness_histogram.png", caption="Histogram of sensor missingness frequencies")
    with col_q2:
        st.subheader("Top features by missingness")
        missing_df = pd.DataFrame(quality["features_by_missingness"])
        st.dataframe(missing_df.head(15), use_container_width=True, hide_index=True)

    st.subheader("Outlier treatment strategy")
    st.markdown(
        """
        - **3-Sigma Capping (Winsorising)**: Extreme readings are capped at $\\mu \\pm 3\\sigma$, bounding
          extreme excursions without deleting rows or creating missing gaps.
        - **IQR Prohibited**: The student manual explicitly bans IQR for SECOM because heavy sensor skewness
          causes IQR thresholds to clip genuine process signals.
        """
    )

    show_lesson(quality["lesson"])


# =========================================================
# STEP 3 — Preprocessing and Step Impact Protocol (SIP)
# =========================================================

st.divider()
st.header("Step 3 — Preprocessing and Step Impact Protocol (SIP)")

st.markdown(
    """
    The data is partitioned using a **Stratified 80/20 train/test split**.
    Transformations and statistics are fitted strictly on training data to prevent **data leakage**.
    """
)

if st.button("Run Step 3: Preprocessing plan & SIP Log", key="step3"):
    run_stage("preprocessing summary", "/case-study/preprocessing")
    run_stage("SIP audit log", "/case-study/sip-log")
    run_stage("reduction summary", "/case-study/reduction-summary")
    run_stage("selected features", "/case-study/selected-features")

preprocessing = get_stage("/case-study/preprocessing")

if preprocessing:
    split_rows = [
        {
            "Dataset": "Training (80%)",
            "Rows": preprocessing["train_rows"],
            "Pass": preprocessing["train_passes"],
            "Fault": preprocessing["train_faults"],
        },
        {
            "Dataset": "Held-Out Test (20%)",
            "Rows": preprocessing["test_rows"],
            "Pass": preprocessing["test_passes"],
            "Fault": preprocessing["test_faults"],
        },
    ]

    st.subheader("1. Train/Test Split (Stratified, random_state=42)")
    st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

    # Dimensionality reduction summary
    red_data = get_stage("/case-study/reduction-summary")
    if red_data and "records" in red_data:
        st.subheader("2. Feature Reduction Progression (590 -> 450 Sensors)")
        st.dataframe(pd.DataFrame(red_data["records"]), use_container_width=True, hide_index=True)

    # Selected Boruta features
    sel_data = get_stage("/case-study/selected-features")
    if sel_data and "features" in sel_data:
        st.subheader("3. Boruta Shadow-Feature Selection (Top 20 Sensors)")
        st.caption(f"Selected {len(sel_data['features'])} parsimonious sensors beating shadow noise:")
        st.markdown(" • ".join([f"`{s}`" for s in sel_data["features"]]))

        c_j1, c_j2 = st.columns(2)
        c_j1.metric("5-Fold CV Recall (450 features)", "2.43%")
        c_j2.metric("5-Fold CV Recall (20 features)", "15.66%", "+544% relative gain")

    # Sensor correlation heatmap
    st.subheader("4. Sensor Correlation Heatmap")
    display_figure("sensor_correlation_heatmap.png", caption="Pairwise correlations across Boruta selected sensors")

    # Step Impact Protocol (SIP) table
    sip_data = get_stage("/case-study/sip-log")
    if sip_data and "records" in sip_data:
        st.subheader("5. Step Impact Protocol (SIP) 4-Column Audit Log")
        st.caption("Records every design decision, pros/cons, and empirical consequences per the manual:")
        sip_df = pd.DataFrame(sip_data["records"])
        st.dataframe(sip_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download SIP Log (CSV)",
            data=sip_df.to_csv(index=False).encode("utf-8"),
            file_name="secom_sip_log.csv",
            mime="text/csv",
            key="download_sip_csv",
        )

    show_lesson(
        "Zero test leakage: all scalers, 3-sigma capping bounds, kNN imputation graphs, "
        "and SMOTE balancing are fitted strictly inside training folds."
    )


# =========================================================
# STEP 4 — Model experiments
# =========================================================

st.divider()
st.header("Step 4 — Compare machine-learning models")

st.markdown(
    """
    Five candidate algorithms were benchmarked across **Stratified 5-Fold Cross Validation**
    with SMOTE applied inside each training fold.
    """
)

if st.button("Run Step 4: Load model comparisons & tuning", key="step4"):
    run_stage("historical model comparison", "/case-study/experiments")
    run_stage("hyperparameter grid search", "/case-study/grid-search")

experiments = get_stage("/case-study/experiments")

if experiments:
    experiment_df = pd.DataFrame(experiments["results"])

    st.subheader("Candidate Model Comparison (Stratified 5-Fold CV)")
    st.dataframe(experiment_df, use_container_width=True, hide_index=True)

    # Plotly bar chart
    fig = px.bar(
        experiment_df,
        x="model",
        y="mean_recall",
        error_y="std_recall" if "std_recall" in experiment_df.columns else None,
        color="model",
        text_auto=".2%",
        title="Candidate Models: Minority Class Recall (5-Fold CV)",
        labels={"model": "Model", "mean_recall": "Mean Recall"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hyperparameter tuning results
    grid_data = get_stage("/case-study/grid-search")
    if grid_data:
        st.subheader("Random Forest Hyperparameter Tuning (GridSearchCV)")
        col_g1, col_g2, col_g3 = st.columns(3)
        col_g1.metric("Optimal max_depth", "10")
        col_g2.metric("Optimal min_samples_leaf", "2")
        col_g3.metric("Tuned 5-Fold CV Recall", f"{grid_data.get('best_cv_recall', 0.3257):.2%}", "+108% over baseline")

    show_lesson(experiments["lesson"])


# =========================================================
# STEP 5 — Threshold and cost analysis
# =========================================================

st.divider()
st.header("Step 5 — Decision threshold and cost analysis")

st.markdown(
    """
    The threshold converts a continuous model probability score into an inspection flag:
    - Score $\\ge$ threshold $\\to$ **FLAG FOR INSPECTION**
    - Score $<$ threshold $\\to$ **NOT FLAGGED**

    In wafer fabrication, costs are asymmetric:
    $$\\text{Total Cost} = 100 \\times \\text{FP} + 400 \\times \\text{FN}$$
    """
)

if st.button("Run Step 5: Load threshold analysis", key="step5"):
    run_stage("threshold analysis", "/case-study/threshold")

threshold_data = get_stage("/case-study/threshold")

if threshold_data:
    st.metric(
        "Cost-Minimizing Decision Threshold",
        f'{threshold_data["selected_threshold"]:.3f}',
        "Optimized on Training OOF CV",
    )

    t_curve = threshold_data.get("threshold_curve", [])
    if t_curve:
        t_df = pd.DataFrame(t_curve)
        fig_cost = px.line(
            t_df.sort_values("threshold"),
            x="threshold",
            y="total_cost",
            title="Total Inspection Cost ($100*FP + $400*FN) vs. Threshold",
            labels={"threshold": "Decision Threshold", "total_cost": "Total Inspection Cost ($)"},
        )
        fig_cost.add_vline(
            x=threshold_data["selected_threshold"],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Optimal: {threshold_data['selected_threshold']:.3f}",
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # Simulator
        st.subheader("Interactive Threshold Simulator")
        sim_val = st.slider("Test Threshold Cutoff", min_value=0.10, max_value=0.90, value=0.495, step=0.005)
        closest_idx = (t_df["threshold"] - sim_val).abs().idxmin()
        sim_row = t_df.loc[closest_idx]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Simulated Recall", f"{sim_row['sensitivity_recall']:.1%}")
        s2.metric("Simulated Specificity", f"{sim_row['specificity']:.1%}")
        s3.metric("Simulated Total Cost", f"${sim_row['total_cost']:,.0f}")
        s4.metric("Simulated Mfg Impact", f"€{sim_row['manufacturing_cost_eur']:,.0f}")

    show_lesson(threshold_data["lesson"])


# =========================================================
# STEP 6 — Final model evaluation
# =========================================================

st.divider()
st.header("Step 6 — Evaluate the final Random Forest")

st.markdown(
    """
    Final evaluation on the **314 held-out wafers** (293 Pass, 21 Defect)
    using the cost-optimal threshold of **0.495**.
    """
)

if st.button("Run Step 6: Load final model evaluation", key="step6"):
    run_stage("final model evaluation", "/case-study/final-model")

final_model = get_stage("/case-study/final-model")

if final_model:
    cm = final_model.get("confusion_matrix", {})

    st.subheader("Holdout Confusion Matrix (314 Wafers, 21 Faults)")

    col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
    col_cm1.metric("True Negatives (TN)", cm.get("true_negatives", 272), "Passed without alarm")
    col_cm2.metric("False Positives (FP)", cm.get("false_positives", 21), "False alarm (cost $100 ea)")
    col_cm3.metric("False Negatives (FN)", cm.get("false_negatives", 13), "Missed fault (cost $400 ea)")
    col_cm4.metric("True Positives (TP)", cm.get("true_positives", 8), "Fault caught & quarantined")

    confusion_df = pd.DataFrame(
        [
            {"Actual Class": "Actual Pass (293)", "Predicted Pass": f"{cm.get('true_negatives', 272)} (TN)", "Predicted Defect": f"{cm.get('false_positives', 21)} (FP)"},
            {"Actual Class": "Actual Defect (21)", "Predicted Pass": f"{cm.get('false_negatives', 13)} (FN)", "Predicted Defect": f"{cm.get('true_positives', 8)} (TP)"},
        ]
    )
    st.dataframe(confusion_df, use_container_width=True, hide_index=True)

    st.subheader("Performance Scorecard")
    metrics = final_model.get("metrics", {})
    metric_cols1 = st.columns(5)
    metric_cols1[0].metric("Recall / Sensitivity", f"{metrics.get('recall', 0.3810):.1%}")
    metric_cols1[1].metric("Specificity", f"{metrics.get('specificity', 0.9283):.1%}")
    metric_cols1[2].metric("Precision", f"{metrics.get('precision', 0.2759):.1%}")
    metric_cols1[3].metric("Accuracy", f"{metrics.get('accuracy', 0.8917):.1%}")
    metric_cols1[4].metric("ROC-AUC", f"{metrics.get('roc_auc', 0.7502):.4f}")

    metric_cols2 = st.columns(5)
    metric_cols2[0].metric("F1 Score", f"{metrics.get('f1', 0.3200):.1%}")
    metric_cols2[1].metric("F2 Score", f"{metrics.get('f2', 0.3540):.1%}")
    metric_cols2[2].metric("PR-AUC", f"{metrics.get('average_precision', 0.2056):.4f}")
    metric_cols2[3].metric("Total Inspection Cost", f"${metrics.get('total_cost', 7300):,}")
    metric_cols2[4].metric("Cost Saved vs Naive", "$1,100", "vs $8,400 all-pass")

    st.subheader("Training vs. Held-Out Test Generalization")
    comp_records = final_model.get("training_vs_test", [])
    if comp_records:
        st.dataframe(pd.DataFrame(comp_records), use_container_width=True, hide_index=True)

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.subheader("ROC Curve")
        display_figure("roc_curve.png", caption="ROC Curve on Held-Out Test Set (AUC = 0.7502)")
    with col_fig2:
        st.subheader("Calibration Diagram")
        display_figure("lesson23_reliability_diagrams.png", caption="Reliability & Calibration Diagrams")

    show_lesson(final_model["lesson"])


# =========================================================
# OPTION — Load data directly from UCI
# =========================================================

st.divider()
st.header("Use the official UCI dataset directly")

st.markdown(
    """
    Don't have a CSV ready? Load the original SECOM dataset from UCI.
    The backend downloads and caches the source files, then lets you choose
    one observation for prediction.
    """
)

st.markdown(
    "[Official SECOM source page](https://archive.ics.uci.edu/dataset/179/secom)"
)

if st.button("Load dataset from UCI", type="primary", key="load_uci"):
    with st.spinner("Downloading and validating the source dataset..."):
        try:
            source_summary = get_api("/source-data/summary")
            st.session_state["uci:summary"] = source_summary
            st.session_state["uci:loaded"] = True
        except requests.RequestException as exc:
            st.error(f"Could not load UCI data: {exc}")

source_summary = st.session_state.get("uci:summary")

if source_summary:
    st.success("UCI dataset loaded successfully.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source observations", f"{source_summary['rows']:,}")
    c2.metric("Sensor features", source_summary["features"])
    c3.metric("Pass samples", f"{source_summary['pass_count']:,}")
    c4.metric("Fault samples", source_summary["fault_count"])

    try:
        full_dataset = requests.get(
            f"{API_URL}/source-data/download-csv",
            timeout=120,
        )
        full_dataset.raise_for_status()

        st.download_button(
            "Download complete UCI dataset as CSV",
            data=full_dataset.content,
            file_name="secom_uci_dataset.csv",
            mime="text/csv",
            key="download_uci_full",
        )
    except Exception as exc:
        st.warning(f"Could not prepare the full CSV download: {exc}")

    st.subheader("Select a source observation")

    row_index = st.number_input(
        "Source row index (starts at 0)",
        min_value=0,
        max_value=int(source_summary["rows"]) - 1,
        value=238,
        step=1,
        key="uci_row_index",
    )

    if st.button("Predict selected source observation", key="predict_uci_row"):
        with st.spinner("Running the saved Random Forest model..."):
            try:
                result = post_api(f"/predict-source-row/{int(row_index)}", {})
                st.session_state["uci:last_prediction"] = result
            except requests.RequestException as exc:
                st.error(f"Source prediction failed: {exc}")

uci_result = st.session_state.get("uci:last_prediction")

if uci_result:
    st.subheader(f'Source observation {uci_result["row_index"]}: prediction results')

    results_df = pd.DataFrame(
        [
            {"Metric": "Predicted class", "Value": uci_result["prediction"]},
            {"Metric": "Model score (%)", "Value": uci_result["risk_percent"]},
            {"Metric": "Decision threshold", "Value": uci_result["threshold"]},
            {"Metric": "Inspection decision", "Value": uci_result["decision"]},
            {"Metric": "Known source label", "Value": uci_result["actual_label"]},
            {
                "Metric": "Prediction matches source label",
                "Value": "Yes" if uci_result["matches_source_label"] else "No",
            },
            {
                "Metric": "Missing sensor readings",
                "Value": uci_result["missing_readings"],
            },
        ]
    )

    st.dataframe(results_df, use_container_width=True, hide_index=True)

    if uci_result["decision"] == "FLAG FOR INSPECTION":
        st.error(f"🚨 **Decision**: {uci_result['decision']} (Score: {uci_result['score']:.4f} >= {uci_result['threshold']})")
    else:
        st.success(f"✅ **Decision**: {uci_result['decision']} (Score: {uci_result['score']:.4f} < {uci_result['threshold']})")


# =========================================================
# STEP 7 — Upload a complete CSV and detect faulty samples
# =========================================================

st.divider()
st.header("Step 7 — Upload a dataset and find potential faults")

st.markdown(
    """
    Upload a complete sensor dataset. The app automatically validates the
    columns, applies the saved model, and returns every observation flagged
    for inspection. No manual preprocessing or threshold configuration is needed.
    """
)

st.caption(
    "Accepted formats: a CSV with sensor_0 ... sensor_589 column names, "
    "or the original headerless, whitespace-separated UCI SECOM sensor file."
)

uploaded_file = st.file_uploader(
    "Upload the complete sensor dataset",
    type=["csv", "data", "txt"],
    key="batch_sensor_file",
)

if uploaded_file is not None:
    st.write(f"**Selected file:** {uploaded_file.name}")
    st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")

    if st.button("Analyze entire dataset", type="primary", key="analyze_batch"):
        try:
            with st.spinner("Validating sensor columns and predicting every observation..."):
                uploaded_file.seek(0)
                response = requests.post(
                    f"{API_URL}/predict-csv",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=300,
                )
                if not response.ok:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"Analysis failed: {detail}")
                else:
                    batch_result = response.json()
                    st.session_state["batch:last_result"] = batch_result
        except requests.Timeout:
            st.error("The analysis timed out.")
        except requests.RequestException as exc:
            st.error(f"Could not connect to FastAPI: {exc}")

batch_result = st.session_state.get("batch:last_result")

if batch_result:
    st.subheader("Analysis summary")

    metrics = st.columns(4)
    metrics[0].metric("Rows processed", batch_result["rows_processed"])
    metrics[1].metric("Flagged for inspection", batch_result["fault_count"])
    metrics[2].metric("Not flagged", batch_result["pass_count"])
    metrics[3].metric("Flagged rate", f'{batch_result["fault_rate_percent"]:.2f}%')

    all_results = pd.DataFrame(batch_result["results"])
    faulty_results = pd.DataFrame(batch_result["faulty_samples"])

    st.subheader("Potentially faulty samples")

    if faulty_results.empty:
        st.success("No samples crossed the configured threshold.")
    else:
        st.warning(f'{len(faulty_results)} observations were flagged. Prioritize these for inspection.')
        st.dataframe(faulty_results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download flagged samples (CSV)",
            data=faulty_results.to_csv(index=False).encode("utf-8"),
            file_name="secom_flagged_samples.csv",
            mime="text/csv",
            key="download_faulty_samples",
        )

    st.subheader("Prediction distribution")
    counts = pd.DataFrame({
        "Prediction": ["Fault", "Pass"],
        "Count": [batch_result["fault_count"], batch_result["pass_count"]],
    })
    fig = px.bar(counts, x="Prediction", y="Count", color="Prediction", text="Count", title="Predicted classes across uploaded dataset")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View predictions for every row"):
        st.dataframe(all_results, use_container_width=True, hide_index=True)

    st.download_button(
        "Download all predictions (CSV)",
        data=all_results.to_csv(index=False).encode("utf-8"),
        file_name="secom_all_predictions.csv",
        mime="text/csv",
        key="download_all_predictions",
    )

    if st.button("Clear batch results", key="clear_batch"):
        st.session_state.pop("batch:last_result", None)
        st.rerun()

st.divider()
st.caption(
    "SECOM Semiconductor Fault Detection Case Study • Compliant with CRISP-DM methodology & Student Manual specifications."
)
