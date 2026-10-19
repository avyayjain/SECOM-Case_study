
import io
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import os

API_URL = st.secrets.get("API_URL")

st.set_page_config(
    page_title="SECOM Case Study",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 SECOM Fault Detection — Case Study")
st.caption(
    "From raw semiconductor sensor data to a machine-learning inspection decision"
)

st.warning(
    "Educational prototype only. Model scores are not guaranteed real-world fault "
    "probabilities. Do not use these predictions to release or reject production "
    "material without independent validation."
)


# ---------------------------------------------------------
# API helpers
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


def show_table(data, key_column=None):
    """Display dictionaries as a readable table."""
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
        else:
            st.error("Backend reports an error")
            st.json(health)
    except requests.RequestException:
        st.error("FastAPI is not reachable")

    st.caption(f"Backend: {API_URL}")

    st.divider()
    st.subheader("Case-study stages")

    st.markdown(
        """
        1. Dataset overview
        2. Data quality and EDA
        3. Preprocessing
        4. Model experiments
        5. Threshold and cost
        6. Final model evaluation
        7. Predict new readings
        """
    )

    if st.button("Clear displayed results", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("stage:") or key.startswith("prediction:"):
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
    cols[0].metric("Observations", overview["observations"])
    cols[1].metric("Sensor features", overview["features"])
    cols[2].metric("Pass samples", overview["pass_count"])
    cols[3].metric("Fault samples", overview["fault_count"])

    st.subheader("Class distribution")

    distribution = pd.DataFrame(
        {
            "Class": ["Pass", "Fault"],
            "Samples": [
                overview["pass_count"],
                overview["fault_count"],
            ],
            "Percentage": [
                overview["pass_percent"],
                overview["fault_percent"],
            ],
        }
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(distribution, use_container_width=True, hide_index=True)

    with col2:
        fig = px.bar(
            distribution,
            x="Class",
            y="Samples",
            color="Class",
            text="Samples",
            title="Pass vs fault samples",
        )
        st.plotly_chart(fig, use_container_width=True)

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
    cols[2].metric(
        "Entirely missing features",
        quality["features_entirely_missing"],
    )
    cols[3].metric("Duplicate rows", quality["duplicate_rows"])

    st.subheader("Feature-level missing values")

    missing_df = pd.DataFrame(quality["features_by_missingness"])

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        missing_df.head(25),
        x="feature",
        y="missing_percent",
        title="Top 25 features by missing-value percentage",
        labels={
            "feature": "Sensor feature",
            "missing_percent": "Missing values (%)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    show_lesson(quality["lesson"])


# =========================================================
# STEP 3 — Preprocessing and experimental design
# =========================================================

st.divider()
st.header("Step 3 — Preprocessing and experimental design")

st.markdown(
    """
    The next step separates the available observations into training and test
    data. The training set is used to fit the model; the held-out test set is
    used for evaluation.

    **Data leakage** happens when information from the test data influences
    training or model selection. Imputation and other learned transformations
    should be fitted using training data only.
    """
)

if st.button("Run Step 3: Preprocessing plan", key="step3"):
    run_stage("preprocessing summary", "/case-study/preprocessing")

preprocessing = get_stage("/case-study/preprocessing")

if preprocessing:
    split_rows = [
        {
            "Dataset": "Training",
            "Rows": preprocessing["train_rows"],
            "Pass": preprocessing["train_passes"],
            "Fault": preprocessing["train_faults"],
            "Missing cells": preprocessing["train_missing_cells"],
        },
        {
            "Dataset": "Test",
            "Rows": preprocessing["test_rows"],
            "Pass": preprocessing["test_passes"],
            "Fault": preprocessing["test_faults"],
            "Missing cells": preprocessing["test_missing_cells"],
        },
    ]

    st.subheader("Train/test split")
    st.dataframe(
        pd.DataFrame(split_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Preprocessing decisions")
    show_table(
        {
            "Split": preprocessing["split_method"],
            "Random state": preprocessing["random_state"],
            "Imputation": preprocessing["imputation"],
            "Leakage prevention": preprocessing["leakage_prevention"],
        }
    )

    show_lesson(preprocessing["lesson"])


# =========================================================
# STEP 4 — Model experiments
# =========================================================

st.divider()
st.header("Step 4 — Compare machine-learning models")

st.markdown(
    """
    Different models make different trade-offs. For fault detection, accuracy
    is not enough: average precision, recall, precision, and F1 are also useful.

    These are the results from the earlier nested cross-validation experiments.
    Loading this step does **not** retrain the models.
    """
)

if st.button("Run Step 4: Load model comparisons", key="step4"):
    run_stage("historical model comparison", "/case-study/experiments")

experiments = get_stage("/case-study/experiments")

if experiments:
    experiment_df = pd.DataFrame(experiments["results"])

    st.dataframe(
        experiment_df,
        use_container_width=True,
        hide_index=True,
    )

    metric = st.selectbox(
        "Compare models using",
        [
            "average_precision",
            "roc_auc",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        ],
        format_func=lambda value: value.replace("_", " ").title(),
        key="experiment_metric",
    )

    fig = px.bar(
        experiment_df,
        x="model",
        y=metric,
        color="model",
        text_auto=".3f",
        title=f"Model comparison: {metric.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(experiments["important_note"])
    show_lesson(experiments["lesson"])


# =========================================================
# STEP 5 — Threshold and cost analysis
# =========================================================

st.divider()
st.header("Step 5 — Decision threshold and cost analysis")

st.markdown(
    """
    The threshold converts a model score into an inspection flag:

    - Score at or above the threshold → flag for inspection.
    - Score below the threshold → not flagged.

    Lowering the threshold can catch more faults, but can also increase false
    alarms. The appropriate balance depends on the real costs of each error.
    """
)

if st.button("Run Step 5: Load threshold analysis", key="step5"):
    run_stage("threshold analysis", "/case-study/threshold")

threshold_data = get_stage("/case-study/threshold")

if threshold_data:
    st.metric(
        "Configured decision threshold",
        f'{threshold_data["selected_threshold"]:.3f}',
    )

    st.subheader("Illustrative false-negative cost scenarios")
    show_table(threshold_data["cost_scenarios"])

    st.caption(threshold_data["threshold_source"])
    show_lesson(threshold_data["lesson"])


# =========================================================
# STEP 6 — Final model evaluation
# =========================================================

st.divider()
st.header("Step 6 — Evaluate the final Random Forest")

st.markdown(
    """
    This stage describes the saved model and its previous holdout evaluation.
    The test results are historical results from the case study, not a fresh
    independent validation.
    """
)

if st.button("Run Step 6: Load final model evaluation", key="step6"):
    run_stage("final model evaluation", "/case-study/final-model")

final_model = get_stage("/case-study/final-model")

if final_model:
    st.subheader("Saved model metrics")

    saved_metrics = final_model.get("saved_test_metrics", {})
    if saved_metrics:
        show_table(saved_metrics)
    else:
        st.caption(
            "The saved metrics file was empty or did not contain a readable table. "
            "The recorded case-study results are shown below."
        )

    cm = final_model["known_test_results"]

    st.subheader("Recorded holdout confusion matrix")

    confusion = pd.DataFrame(
        [
            {
                "Actual class": "Pass",
                "Predicted Pass": cm["true_pass"],
                "Predicted Fault": cm["false_alarm"],
            },
            {
                "Actual class": "Fault",
                "Predicted Pass": cm["missed_fault"],
                "Predicted Fault": cm["detected_fault"],
            },
        ]
    )

    st.dataframe(confusion, use_container_width=True, hide_index=True)

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Average precision",
        f'{cm["average_precision"]:.3f}',
    )
    metric_cols[1].metric("Precision", f'{cm["precision"]:.3f}')
    metric_cols[2].metric("Recall", f'{cm["recall"]:.3f}')
    metric_cols[3].metric("F1", f'{cm["f1"]:.3f}')

    st.error(
        f'On the recorded test split, {cm["missed_fault"]} faults were missed '
        f'and {cm["false_alarm"]} passing samples were flagged.'
    )

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
    c1.metric("Source observations", source_summary["rows"])
    c2.metric("Sensor features", source_summary["features"])
    c3.metric("Pass samples", source_summary["pass_count"])
    c4.metric("Fault samples", source_summary["fault_count"])

    st.caption(
        f'Missing sensor cells: {source_summary["missing_cells"]:,}. '
        "Missing sensor readings will be handled by the saved model pipeline."
    )

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

    except requests.RequestException as exc:
        st.warning(f"Could not prepare the full CSV download: {exc}")

    st.subheader("Select a source observation")

    row_index = st.number_input(
        "Source row index (starts at 0)",
        min_value=0,
        max_value=int(source_summary["rows"]) - 1,
        value=0,
        step=1,
        key="uci_row_index",
    )

    try:
        row_response = requests.get(
            f"{API_URL}/source-data/download-row/{int(row_index)}",
            timeout=60,
        )
        row_response.raise_for_status()

        st.download_button(
            "Download selected observation as CSV",
            data=row_response.content,
            file_name=f"secom_source_row_{int(row_index)}.csv",
            mime="text/csv",
            key=f"download_uci_row_{int(row_index)}",
        )

    except requests.RequestException as exc:
        st.warning(f"Could not prepare this row for download: {exc}")

    if st.button(
        "Predict selected source observation",
        key="predict_uci_row",
    ):
        with st.spinner("Running the saved Random Forest model..."):
            try:
                result = post_api(
                    f"/predict-source-row/{int(row_index)}",
                    {},
                )
                st.session_state["uci:last_prediction"] = result

            except requests.RequestException as exc:
                st.error(f"Source prediction failed: {exc}")

uci_result = st.session_state.get("uci:last_prediction")

if uci_result:
    st.subheader(
        f'Source observation {uci_result["row_index"]}: prediction results'
    )

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

    if uci_result["matches_source_label"]:
        st.success("Prediction matches this observation's source label.")
    else:
        st.warning("Prediction differs from this observation's source label.")

    st.caption(uci_result["note"])



# =========================================================
# STEP 7 — Upload readings and predict
# =========================================================

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

    if st.button(
        "Analyze entire dataset",
        type="primary",
        key="analyze_batch",
    ):
        try:
            with st.spinner(
                "Validating sensor columns and predicting every observation..."
            ):
                uploaded_file.seek(0)

                response = requests.post(
                    f"{API_URL}/predict-csv",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv",
                        )
                    },
                    timeout=300,
                )

                if not response.ok:
                    try:
                        detail = response.json().get(
                            "detail", response.text
                        )
                    except ValueError:
                        detail = response.text

                    st.error(f"Analysis failed: {detail}")
                else:
                    batch_result = response.json()
                    st.session_state["batch:last_result"] = batch_result

        except requests.Timeout:
            st.error(
                "The analysis timed out. Try a smaller file or increase the "
                "API timeout for large datasets."
            )
        except requests.RequestException as exc:
            st.error(
                f"Could not connect to the FastAPI backend: {exc}"
            )


# Keep results visible when Streamlit reruns.
batch_result = st.session_state.get("batch:last_result")

if batch_result:
    st.subheader("Analysis summary")

    metrics = st.columns(4)
    metrics[0].metric("Rows processed", batch_result["rows_processed"])
    metrics[1].metric("Flagged for inspection", batch_result["fault_count"])
    metrics[2].metric("Not flagged", batch_result["pass_count"])
    metrics[3].metric(
        "Flagged rate",
        f'{batch_result["fault_rate_percent"]:.2f}%',
    )

    st.caption(
        f'Features used: {batch_result["features_used"]} | '
        f'Threshold: {batch_result["threshold"]:.3f} | '
        f'Missing sensor readings: {batch_result["missing_readings_total"]:,}'
    )

    all_results = pd.DataFrame(batch_result["results"])
    faulty_results = pd.DataFrame(batch_result["faulty_samples"])

    st.subheader("Potentially faulty samples")

    if faulty_results.empty:
        st.success(
            "No samples crossed the configured threshold. This does not "
            "guarantee that the dataset contains no faults."
        )
    else:
        st.warning(
            f'{len(faulty_results)} observations were flagged. '
            "Prioritize these for further inspection."
        )

        st.dataframe(
            faulty_results,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download flagged samples",
            data=faulty_results.to_csv(index=False).encode("utf-8"),
            file_name="secom_flagged_samples.csv",
            mime="text/csv",
            key="download_faulty_samples",
        )

    st.subheader("Prediction distribution")

    counts = pd.DataFrame({
        "Prediction": ["Fault", "Pass"],
        "Count": [
            batch_result["fault_count"],
            batch_result["pass_count"],
        ],
    })

    fig = px.bar(
        counts,
        x="Prediction",
        y="Count",
        color="Prediction",
        text="Count",
        title="Predicted classes across the uploaded dataset",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View predictions for every row"):
        st.dataframe(
            all_results,
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download all predictions",
        data=all_results.to_csv(index=False).encode("utf-8"),
        file_name="secom_all_predictions.csv",
        mime="text/csv",
        key="download_all_predictions",
    )

    st.info(batch_result["note"])

    if st.button("Clear batch results", key="clear_batch"):
        st.session_state.pop("batch:last_result", None)
        st.rerun()
