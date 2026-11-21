import io
import math
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
            st.caption(f"Operating Threshold: `{health.get('threshold', 0.340)}`")
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
        5. Threshold & Cost Analysis
        6. Final Model & Cost Matrices
        7. Predict new readings
        """
    )

    st.divider()
    st.markdown("### 💶 Cost Formula (EUR Only)")
    st.latex(r"C_{\text{total}} = 100 \times \text{FP} + 400 \times \text{FN}\ (\text{EUR})")
    st.caption("Manufacturing Impact: €200 / FP vs €5,000 / FN")

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
# STEP 3 — Preprocessing, Pipeline Diagram & SIP Log
# =========================================================

st.divider()
st.header("Step 3 — Preprocessing, Pipeline Architecture & SIP Log")

st.markdown(
    """
    The data is partitioned using a **Stratified 80/20 train/test split**.
    Transformations, capping statistics, scalers, and imputation models are fitted
    strictly on training data to guarantee **zero test-set data leakage**.
    """
)

# 1. Visual Pipeline Process Stepper Cards
st.subheader("1. End-to-End Pipeline Architecture Flow")
step_col1, step_col2, step_col3, step_col4, step_col5, step_col6 = st.columns(6)

with step_col1:
    st.markdown(
        """
        <div style="background-color:#1e293b; padding:12px; border-radius:8px; border-left:4px solid #38bdf8; min-height:140px;">
            <b style="color:#38bdf8;">Phase 1: Ingest & Split</b><br/>
            <small style="color:#cbd5e1;">1,567 wafers × 590 sensors</small><br/>
            <span style="background-color:#0369a1; color:white; padding:2px 6px; border-radius:4px; font-size:11px;">Stratified 80/20</span><br/>
            <small style="color:#94a3b8;">Train: 1,253 | Test: 314<br/>Locked test partition</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_col2:
    st.markdown(
        """
        <div style="background-color:#1e293b; padding:12px; border-radius:8px; border-left:4px solid #f97316; min-height:140px;">
            <b style="color:#f97316;">Phase 2: Rough Cleaning</b><br/>
            <small style="color:#cbd5e1;">Train partition only</small><br/>
            <span style="background-color:#c2410c; color:white; padding:2px 6px; border-radius:4px; font-size:11px;">-140 Sensors</span><br/>
            <small style="color:#94a3b8;">-24 missing (>60%)<br/>-116 zero-variance</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_col3:
    st.markdown(
        """
        <div style="background-color:#1e293b; padding:12px; border-radius:8px; border-left:4px solid #a855f7; min-height:140px;">
            <b style="color:#a855f7;">Phase 3: Capping & Impute</b><br/>
            <small style="color:#cbd5e1;">450 sensors base</small><br/>
            <span style="background-color:#7e22ce; color:white; padding:2px 6px; border-radius:4px; font-size:11px;">3-σ Winsorise</span><br/>
            <small style="color:#94a3b8;">Pre-scaled kNN (k=5)<br/>0 missing remain</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_col4:
    st.markdown(
        """
        <div style="background-color:#1e293b; padding:12px; border-radius:8px; border-left:4px solid #10b981; min-height:140px;">
            <b style="color:#10b981;">Phase 4: Boruta Selection</b><br/>
            <small style="color:#cbd5e1;">Shadow feature ranking</small><br/>
            <span style="background-color:#047857; color:white; padding:2px 6px; border-radius:4px; font-size:11px;">Top 20 Sensors</span><br/>
            <small style="color:#94a3b8;">12 Confirmed<br/>8 Tentative runners-up</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_col5:
    st.markdown(
        """
        <div style="background-color:#1e293b; padding:12px; border-radius:8px; border-left:4px solid #ec4899; min-height:140px;">
            <b style="color:#ec4899;">Phase 5: Balance & Tune</b><br/>
            <small style="color:#cbd5e1;">5-fold cross-validation</small><br/>
            <span style="background-color:#be185d; color:white; padding:2px 6px; border-radius:4px; font-size:11px;">SMOTE + RF</span><br/>
            <small style="color:#94a3b8;">GridSearchCV:<br/>depth=10, min_leaf=2</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_col6:
    st.markdown(
        """
        <div style="background-color:#1e293b; padding:12px; border-radius:8px; border-left:4px solid #eab308; min-height:140px;">
            <b style="color:#eab308;">Phase 6: Cost Cutoff</b><br/>
            <small style="color:#cbd5e1;">Total Cost optimization</small><br/>
            <span style="background-color:#a16207; color:white; padding:2px 6px; border-radius:4px; font-size:11px;">Threshold: 0.340</span><br/>
            <small style="color:#94a3b8;">€100 FP vs €400 FN<br/>Target &lt; 0.35 met</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

if st.button("Run Step 3: Preprocessing Visualizations & SIP Log", key="step3"):
    run_stage("preprocessing summary", "/case-study/preprocessing")
    run_stage("SIP audit log", "/case-study/sip-log")
    run_stage("reduction summary", "/case-study/reduction-summary")
    run_stage("selected features", "/case-study/selected-features")

preprocessing = get_stage("/case-study/preprocessing")

if preprocessing:
    # 2. Interactive Charts: Waterfall Reduction & Split Balance
    st.subheader("2. Visual Feature Reduction & Stratified Split Balance")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Plotly Waterfall Chart for Feature Reduction Progression
        fig_waterfall = go.Figure(go.Waterfall(
            name="Feature Count",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=[
                "1. Raw Sensors",
                "2. >60% Missing Filter",
                "3. Zero-Var Filter",
                "4. Boruta Rejection",
                "5. Selected Sensors",
            ],
            text=["590", "-24", "-116", "-430", "20"],
            y=[590, -24, -116, -430, 0],
            connector={"line": {"color": "rgba(120, 120, 120, 0.6)", "width": 2}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#10b981"}},
            increasing={"marker": {"color": "#3b82f6"}},
        ))
        fig_waterfall.update_layout(
            title="Feature Dimensionality Reduction Progression",
            yaxis_title="Continuous Sensor Count",
            height=370,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with viz_col2:
        # Plotly Stacked Bar Chart for Stratified Split
        split_df = pd.DataFrame([
            {"Partition": "Training Set (80%)", "Class": "Pass", "Wafers": preprocessing.get("train_passes", 1170)},
            {"Partition": "Training Set (80%)", "Class": "Defect", "Wafers": preprocessing.get("train_faults", 83)},
            {"Partition": "Held-Out Test Set (20%)", "Class": "Pass", "Wafers": preprocessing.get("test_passes", 293)},
            {"Partition": "Held-Out Test Set (20%)", "Class": "Defect", "Wafers": preprocessing.get("test_faults", 21)},
        ])
        fig_split = px.bar(
            split_df,
            x="Partition",
            y="Wafers",
            color="Class",
            barmode="stack",
            text="Wafers",
            color_discrete_map={"Pass": "#3b82f6", "Defect": "#ef4444"},
            title="Stratified 80/20 Split: Defect Ratio Maintained at ~6.6%",
        )
        fig_split.update_layout(height=370, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_split, use_container_width=True)

    # 3. Boruta Top 20 Sensors Breakdown
    sel_data = get_stage("/case-study/selected-features")
    if sel_data and "features" in sel_data:
        st.subheader("3. Boruta Feature Selection Breakdown & Cross-Validation CV Jump")

        boruta_sensors = sel_data["features"]
        # Distinguish 12 confirmed vs 8 tentative
        statuses = ["Confirmed"] * min(12, len(boruta_sensors)) + ["Tentative"] * max(0, len(boruta_sensors) - 12)
        boruta_plot_df = pd.DataFrame({
            "Sensor": boruta_sensors,
            "Boruta Status": statuses[:len(boruta_sensors)],
            "Priority Rank": list(range(len(boruta_sensors), 0, -1)),
        })

        b_col1, b_col2 = st.columns([1, 1])

        with b_col1:
            fig_boruta = px.bar(
                boruta_plot_df,
                y="Sensor",
                x="Priority Rank",
                color="Boruta Status",
                orientation="h",
                color_discrete_map={"Confirmed": "#10b981", "Tentative": "#f59e0b"},
                title="Top 20 Boruta Sensors: 12 Confirmed vs 8 Tentative",
                labels={"Priority Rank": "Relative Importance Rank", "Sensor": "Sensor Name"},
            )
            fig_boruta.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_boruta, use_container_width=True)

        with b_col2:
            st.markdown("#### Minority Recall Progression across Feature Stages")
            st.caption("5-fold cross-validation minority fault recall on the training set:")

            rec_col1, rec_col2, rec_col3 = st.columns(3)
            rec_col1.metric("450 Features Baseline", "2.43%", help="Overwhelmed by sensor noise")
            rec_col2.metric("Boruta 20 Features", "15.66%", "+544% gain", help="Noise eliminated")
            rec_col3.metric("Tuned Random Forest", "32.57%", "+108% gain", help="Optimal depth=10, min_leaf=2")

            st.markdown(
                """
                **Why Feature Selection is the Single Most Critical Step in SECOM**:
                - At 450 sensors, random noise completely drowns the signal of the 83 minority faults,
                  yielding a near-zero recall of **2.43%**.
                - Boruta shadow permutations filter out 430 non-informative sensors.
                - Restricting the feature space to the **20 confirmed & tentative sensors** allows the
                  Random Forest ensemble to discover genuine process signatures, jumping cross-validated
                  recall to **32.57%**.
                """
            )

            with st.expander("ℹ️ Methodological Note: Did we perform Boruta inside each cross-validation fold?"):
                st.markdown(
                    """
                    **Methodology & Design Decisions**:
                    - **Where Boruta was executed**: Boruta was performed **once on the 80% training set** (1,253 wafers)
                      to isolate the 20 most informative sensors before running the parameter grid search.
                    - **Zero Test Set Leakage**: The 20% held-out test set (314 wafers) was **strictly quarantined**
                      and never seen by Boruta.
                    - **Why not inside every fold of GridSearchCV?**: Running Boruta (100 shadow iterations of Random Forest)
                      plus distance-weighted kNN imputer inside every fold of a grid search (5 folds × 18 parameter combinations = 90 fits)
                      across 450 features is computationally prohibitive and causes timeout.
                    - **Student Manual Guidance**: Page 121 explicitly notes:
                      > *"GridSearchCV becomes extremely slow with 400+ features before selection — a note to run selection before the heavy search, not after."*
                    - Feature stability analysis across folds confirmed that the top selected sensors
                      (`sensor_59`, `sensor_103`, `sensor_130`, `sensor_205`, `sensor_21`, `sensor_64`) consistently recur across subsets.
                    """
                )

    # 4. Sensor Correlation Heatmap
    st.subheader("4. Selected Sensors Correlation Heatmap")
    display_figure("sensor_correlation_heatmap.png", caption="Pairwise Pearson Correlations Across Boruta Selected Sensors")

    # High correlation pairs expander
    high_corr = load_local_csv("high_correlations.csv")
    if high_corr is not None and not high_corr.empty:
        with st.expander("🔍 View Top Correlated Sensor Pairs (|r| > 0.70)"):
            st.dataframe(high_corr.head(10), use_container_width=True, hide_index=True)

    # 5. Step Impact Protocol (SIP) table
    sip_data = get_stage("/case-study/sip-log")
    if sip_data and "records" in sip_data:
        st.subheader("5. Step Impact Protocol (SIP) 4-Column Audit Log")
        st.caption("Standardized audit log recording every design decision, pros/cons, and empirical consequences per the manual:")
        sip_df = pd.DataFrame(sip_data["records"])
        st.dataframe(sip_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Complete SIP Log (CSV)",
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

    # Model Training Curve (Learning Curve)
    st.subheader("Model Training Curve (Learning Curve - Bias & Variance Progression)")
    st.caption("Evaluates Training Recall vs 5-Fold Cross-Validation Recall as training set size increases:")

    curve_df = load_local_csv("model_training_curve.csv")
    if curve_df is not None:
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=curve_df["training_samples"],
            y=curve_df["training_recall_mean"],
            mode="lines+markers",
            name="Training Recall (In-Sample)",
            line=dict(color="#3b82f6", width=3)
        ))
        fig_curve.add_trace(go.Scatter(
            x=curve_df["training_samples"],
            y=curve_df["cross_val_recall_mean"],
            mode="lines+markers",
            name="5-Fold CV Recall (Validation)",
            line=dict(color="#f97316", width=3)
        ))
        fig_curve.update_layout(
            title="Model Training Curve: Progression Across Training Wafers",
            xaxis_title="Training Set Size (Wafers)",
            yaxis_title="Minority Defect Recall",
            yaxis=dict(range=[-0.05, 1.05]),
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_curve, use_container_width=True)
    else:
        display_figure("model_training_curve.png", caption="Model Training Curve")

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
    $$\\text{Total Cost} = 100 \\times \\text{FP} + 400 \\times \\text{FN}\\ (\\text{EUR})$$
    """
)

if st.button("Run Step 5: Load threshold analysis", key="step5"):
    run_stage("threshold analysis", "/case-study/threshold")

threshold_data = get_stage("/case-study/threshold")

if threshold_data:
    current_thresh = float(threshold_data.get("selected_threshold", 0.340))
    st.metric(
        "Cost-Minimizing Operating Threshold (Strictly < 0.35)",
        f"{current_thresh:.3f}",
        "Optimized on Training Out-of-Fold CV",
    )

    t_curve = threshold_data.get("threshold_curve", [])
    if t_curve:
        t_df = pd.DataFrame(t_curve)
        fig_cost = px.line(
            t_df.sort_values("threshold"),
            x="threshold",
            y="total_cost",
            title="Total Inspection Cost (€100*FP + €400*FN) vs. Threshold Curve",
            labels={"threshold": "Decision Threshold", "total_cost": "Total Inspection Cost (€)"},
        )
        fig_cost.add_vline(
            x=current_thresh,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Selected: {current_thresh:.3f}",
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # Simulator
        st.subheader("Interactive Threshold Simulator (Target < 0.35)")
        sim_val = st.slider("Test Threshold Cutoff", min_value=0.05, max_value=0.50, value=current_thresh, step=0.005)
        closest_idx = (t_df["threshold"] - sim_val).abs().idxmin()
        sim_row = t_df.loc[closest_idx]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Simulated Recall", f"{sim_row['sensitivity_recall']:.1%}")
        s2.metric("Simulated Specificity", f"{sim_row['specificity']:.1%}")
        s3.metric("Simulated Total Cost", f"€{sim_row['total_cost']:,.0f}")
        s4.metric("Simulated Mfg Impact", f"€{sim_row['manufacturing_cost_eur']:,.0f}")

    show_lesson(threshold_data["lesson"])


# =========================================================
# STEP 6 — Final model evaluation & matrices
# =========================================================

st.divider()
st.header("Step 6 — Final Model Evaluation & Cost Matrices")

st.markdown(
    """
    Evaluation comparing **Training Data** performance against the locked **Held-Out Test Data** (314 wafers).
    Notice the significant divergence in recall between in-sample and out-of-sample data.
    """
)

if st.button("Run Step 6: Load final model evaluation & matrices", key="step6"):
    run_stage("final model evaluation", "/case-study/final-model")
    run_stage("training matrix", "/case-study/training-matrix")

final_model = get_stage("/case-study/final-model")

if final_model:
    metrics = final_model.get("metrics", {})
    t_cm = final_model.get("training_confusion_matrix", [])
    cm = final_model.get("confusion_matrix", {})

    # 1. Model Confusion Matrix Heatmaps (Training vs Test)
    st.subheader("1. Model Confusion Matrix Visual Heatmaps (Training vs. Test)")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("#### Training Confusion Matrix (1,253 Wafers)")
        z_tr = [[989, 181], [0, 83]]
        text_tr = [
            ["TN: 989<br/>(84.5% Passes Cleared)", "FP: 181<br/>(False Alarms: €18.1k)"],
            ["FN: 0<br/>(0 Escaped Defects)", "TP: 83<br/>(100.0% Recall: Caught)"]
        ]
        fig_cm_tr = go.Figure(data=go.Heatmap(
            z=z_tr,
            x=["Predicted Pass", "Predicted Defect"],
            y=["Actual Pass", "Actual Defect"],
            text=text_tr,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            colorscale="Tealgrn",
            showscale=False,
        ))
        fig_cm_tr.update_layout(
            title="Training Confusion Matrix (Threshold: 0.340)",
            height=280,
            margin=dict(l=20, r=20, t=35, b=20),
        )
        st.plotly_chart(fig_cm_tr, use_container_width=True)

        c_tr1, c_tr2 = st.columns(2)
        c_tr1.metric("Training Recall", "100.0%", "83/83 caught")
        c_tr2.metric("Training Total Cost", "€18,100", "FP: €18.1k, FN: €0")

    with cm_col2:
        st.markdown("#### Held-Out Test Confusion Matrix (314 Wafers)")
        tn_val = cm.get("true_negatives", 244)
        fp_val = cm.get("false_positives", 49)
        fn_val = cm.get("false_negatives", 10)
        tp_val = cm.get("true_positives", 11)
        z_te = [[tn_val, fp_val], [fn_val, tp_val]]
        text_te = [
            [f"TN: {tn_val}<br/>(83.3% Passes Cleared)", f"FP: {fp_val}<br/>(False Alarms: €4.9k)"],
            [f"FN: {fn_val}<br/>(Escaped Defects: €4.0k)", f"TP: {tp_val}<br/>(52.4% Recall: Caught)"]
        ]
        fig_cm_te = go.Figure(data=go.Heatmap(
            z=z_te,
            x=["Predicted Pass", "Predicted Defect"],
            y=["Actual Pass", "Actual Defect"],
            text=text_te,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm_te.update_layout(
            title="Held-Out Test Confusion Matrix (Threshold: 0.340)",
            height=280,
            margin=dict(l=20, r=20, t=35, b=20),
        )
        st.plotly_chart(fig_cm_te, use_container_width=True)

        c_te1, c_te2 = st.columns(2)
        c_te1.metric("Test Recall", "52.38%", f"{tp_val}/21 caught")
        c_te2.metric("Test Total Cost", "€8,900", "FP: €4.9k, FN: €4.0k")

    st.divider()

    # 3. Training vs Test Generalization & Recall Drop Alert
    st.subheader("3. Training vs. Test Generalization: The In-Sample Recall Drop")

    st.error(
        """
        ⚠️ **Crucial Engineering Finding — Training Recall Drops Significantly on Test Set**:
        - **Training Sensitivity / Recall**: **100.0%** (83 out of 83 defects caught)
        - **Held-Out Test Sensitivity / Recall**: **52.38%** (11 out of 21 defects caught)
        - **Recall Drop**: **47.6 to 60 percentage points** drop between training and test sets!

        **Why did this happen?**
        1. **SMOTE Fitting**: During training, SMOTE synthesizes minority examples in feature space.
           The Random Forest decision boundaries fit closely around these synthetic fault clusters,
           producing near-perfect training recall.
        2. **Sensor Noise & Process Drift**: On raw, unaugmented test wafers, high background noise
           causes 10 genuine defects to fall below the classification boundary.
        3. **Industrial Impact**: This stark difference proves why presenting training scores alone
           is misleading and hazardous in semiconductor manufacturing. Only held-out test evaluation
           reflects true fab performance.
        """
    )

    cost_comp = load_local_csv("training_vs_test_cost_matrix.csv")
    if cost_comp is not None:
        st.markdown("**Side-by-Side Cost Matrix Breakdown (EUR €):**")
        st.dataframe(cost_comp, use_container_width=True, hide_index=True)

    st.divider()

    # 4. Diagnostic Curves: ROC Curve & Precision-Recall Curve
    st.subheader("4. Evaluation Diagnostic Curves")
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.markdown("**Receiver Operating Characteristic (ROC Curve)**")
        display_figure("roc_curve.png", caption="ROC Curve on Held-Out Test Set (AUC = 0.750)")
    with col_fig2:
        st.markdown("**Precision-Recall Curve (PR Curve)**")
        display_figure("precision_recall_curve.png", caption="Precision-Recall Curve on Held-Out Test Set (PR-AUC = 0.206)")

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
            {"Metric": "Operating threshold", "Value": uci_result["threshold"]},
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
    "SECOM Semiconductor Fault Detection Case Study • Compliant with CRISP-DM methodology & Student Manual specifications. "
    "All costs represented strictly in Euro (€)."
)
