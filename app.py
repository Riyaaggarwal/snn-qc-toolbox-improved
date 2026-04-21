import base64
import datetime
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import streamlit as st
import torch
import yaml

from snnqc.data_loader import (
    combined_csv_template,
    dataframe_to_csv_bytes,
    dataset_summary,
    labels_csv_template,
    load_builtin_eeg_dataset as _load_builtin_eeg_dataset,
    parse_combined_csv,
    parse_sample_csvs,
    read_uploaded_csv,
    sample_csv_template,
)
from snnqc.plots import feature_layout_figure, raw_and_spike_figure

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score as accuracy,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from neucube import Reservoir
    from neucube.encoder import Delta
    from neucube.sampler import SpikeCount
    from neucube.training import STDP
    from neucube.qfeatures import extract_features
    from neucube.qkernel import kernel_matrix, kernel
except ImportError as e:
    st.error(f"Critical import error: {e}")
    st.info("Ensure the 'neucube' package folder is present alongside app.py.")
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

_config_path = _HERE / "config.yaml"
_cfg = yaml.safe_load(_config_path.read_text()) if _config_path.exists() else {}
_app_cfg = _cfg.get("app", {})
_defaults = _cfg.get("defaults", {})

APP_NAME = _app_cfg.get("name", "SpikeSense Studio")
APP_TAGLINE = _app_cfg.get("tagline", "Time-series classification with spiking features and explainable model results.")

DATASET_STATE_KEYS = [
    "X_raw", "X", "y", "feature_names", "dataset_name", "data_ready",
    "map_initialised", "features_ready", "snn_features",
]
DERIVED_STATE_KEYS = ["map_initialised", "features_ready", "snn_features"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="📈")

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stApp { background: #050505; color: #E5E7EB; }
        .block-container { padding-top: 1.4rem; max-width: 1180px; }
        .snnqc-hero {
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 8px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.15rem;
            background: #111111;
            box-shadow: 0 1px 10px rgba(0,0,0,0.45);
        }
        .snnqc-hero h1 { font-size: 2.1rem; line-height: 1.15; margin: 0 0 0.35rem 0; color: #F8FAFC; }
        .snnqc-hero p  { color: #CBD5E1; font-size: 1rem; margin: 0; }
        .snnqc-eyebrow {
            color: #60A5FA; font-size: 0.78rem; font-weight: 700;
            letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem;
        }
        [data-testid="stMetric"] {
            background: #111111; border: 1px solid rgba(148,163,184,0.18);
            border-radius: 8px; padding: 0.85rem 0.95rem;
        }
        div[data-testid="stExpander"] {
            background: #0B0B0B; border-radius: 8px; border: 1px solid rgba(148,163,184,0.16);
        }
        section[data-testid="stSidebar"] { background: #000000; }
        section[data-testid="stSidebar"] * { color: #e2e8f0; }
        div[data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,0.16); border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="snnqc-hero">
        <div class="snnqc-eyebrow">Predictive signal intelligence</div>
        <h1>{APP_NAME}</h1>
        <p>{APP_TAGLINE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── State helpers ──────────────────────────────────────────────────────────────

def clear_derived_state():
    for key in DERIVED_STATE_KEYS:
        st.session_state.pop(key, None)


def clear_dataset_state():
    for key in DATASET_STATE_KEYS:
        st.session_state.pop(key, None)


# ── Cached data operations ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_builtin_eeg_dataset():
    return _load_builtin_eeg_dataset()


@st.cache_data(show_spinner=False)
def encode_dataset(X_raw, thresh):
    encoder = Delta(threshold=thresh)
    return encoder.encode_dataset(X_raw)


# ── Render helpers ─────────────────────────────────────────────────────────────

def render_csv_templates():
    with st.expander("CSV Templates"):
        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "Combined CSV template", dataframe_to_csv_bytes(combined_csv_template()),
            file_name="snnqc_combined_template.csv", mime="text/csv",
        )
        c2.download_button(
            "Sample CSV template", dataframe_to_csv_bytes(sample_csv_template()),
            file_name="snnqc_sample_template.csv", mime="text/csv",
        )
        c3.download_button(
            "Labels CSV template", dataframe_to_csv_bytes(labels_csv_template()),
            file_name="snnqc_labels_template.csv", mime="text/csv",
        )


def render_dataset_preview(X_raw, y_data, feature_names):
    preview = dataset_summary(X_raw, y_data, feature_names)
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", preview["samples"])
    c2.metric("Timepoints", preview["timepoints"])
    c3.metric("Features", preview["features"])
    st.write("Class balance")
    st.dataframe(
        preview["class_counts"].rename("count").reset_index().rename(columns={"index": "class"}),
        use_container_width=True,
    )


def render_loaded_dataset_summary(X, y, feature_names):
    summary = dataset_summary(X, y, feature_names)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", summary["samples"])
    c2.metric("Timepoints", summary["timepoints"])
    c3.metric("Features", summary["features"])
    c4.metric("Classes", len(summary["class_counts"]))
    st.dataframe(
        summary["class_counts"].rename("count").reset_index().rename(columns={"index": "class"}),
        use_container_width=True,
    )
    with st.expander("Feature names"):
        st.write(", ".join(map(str, feature_names)))


def result_interpretation(accuracy_value, f1_value, y_true, y_pred):
    baseline = pd.Series(y_true).value_counts(normalize=True).max()
    lift = accuracy_value - baseline

    if accuracy_value >= baseline + 0.15 and f1_value >= baseline:
        strength = "The classifier is performing meaningfully above the majority-class baseline."
    elif accuracy_value >= baseline:
        strength = "The classifier is slightly above baseline — treat the result as exploratory."
    else:
        strength = "The classifier is not beating the majority-class baseline yet."

    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true), key=str)
    confusion_note = "No off-diagonal class confusion observed."
    if cm.shape[0] > 1:
        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)
        if off_diag.max() > 0:
            r, c = np.unravel_index(off_diag.argmax(), off_diag.shape)
            confusion_note = (
                f"Most common confusion: true class '{labels[r]}' predicted as "
                f"'{labels[c]}' ({off_diag[r, c]} samples)."
            )

    return {"baseline": baseline, "lift": lift, "strength": strength, "confusion_note": confusion_note}


def push_experiment_result(dataset_name, model_name, n_samples, n_features, acc, f1, seed):
    if "experiment_history" not in st.session_state:
        st.session_state["experiment_history"] = []
    st.session_state["experiment_history"].append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "model": model_name,
        "samples": n_samples,
        "features_used": n_features,
        "accuracy": round(float(acc), 4),
        "weighted_f1": round(float(f1), 4),
        "seed": seed,
    })


def build_experiment_config(threshold_val, res_c, res_l, stdp_pos, stdp_neg,
                             mem_thr_val, svm_c, k_folds, seed):
    config = {
        "app": APP_NAME,
        "dataset": st.session_state.get("dataset_name"),
        "hyperparameters": {
            "delta_threshold": threshold_val,
            "reservoir_c": res_c,
            "reservoir_l": res_l,
            "stdp_a_pos": stdp_pos,
            "stdp_a_neg": stdp_neg,
            "membrane_threshold": mem_thr_val,
            "svm_c": svm_c,
            "cv_folds": k_folds,
            "random_seed": seed,
        },
    }
    if st.session_state.get("data_ready"):
        X = st.session_state["X"]
        y = st.session_state["y"]
        config["data"] = {
            "shape": list(X.shape),
            "classes": [str(v) for v in sorted(pd.Series(y).dropna().unique(), key=str)],
            "feature_names": [str(v) for v in st.session_state.get("feature_names", [])],
        }
    if st.session_state.get("features_ready"):
        config["snn_features"] = {"shape": list(st.session_state["snn_features"].shape)}
    return config


# ── Help page ──────────────────────────────────────────────────────────────────

def render_help_page():
    st.markdown("## Help & Product Info")
    st.write(
        f"{APP_NAME} helps teams explore labelled time-series classification using spiking feature "
        "extraction, classical model baselines, and an optional quantum-kernel comparison."
    )

    tab_ov, tab_data, tab_pipe, tab_res, tab_about = st.tabs(
        ["Overview", "Data Formats", "Pipeline", "Results", "About"]
    )

    with tab_ov:
        st.markdown("""
            ### What this app is for
            Use this app when you have labelled signal, sensor, biomedical, operational, or experimental
            time-series samples and want to test whether spiking features help separate classes.

            The built-in EEG dataset is a reference demo. Uploaded datasets can come from sensors,
            wearables, industrial monitoring, health signals, finance, or any repeated time-series source.

            ### Current scope
            - Works with regularly sampled, fixed-length time-series data.
            - Every sample must have the same number of timepoints.
            - **Quantum Kernel SVM** is an experimental research feature: binary classification,
              exactly two selected features, classical simulation (not real quantum hardware), slow on
              large datasets. Use Classical SVM or Logistic Regression for general use.
            - Classical SVM and Logistic Regression support multiple features and multiple classes.
        """)

    with tab_data:
        st.markdown("""
            ### Single uploaded table
            One row per sample-timepoint. Required columns: sample ID, label, numeric features.
            Optional: a time/order column for sorting.

            ### Multiple sample files
            One CSV per sample (timepoints × features). Upload a separate labels CSV with one label per
            sample file, matched in sorted filename order.

            ### Tips
            - Check the parsed preview for samples, timepoints, feature count, and class balance.
            - Imbalanced classes reduce cross-validation reliability — use Weighted F1 in those cases.
            - Download the CSV templates below to see the expected format.
        """)

    with tab_pipe:
        st.markdown("""
            ### Processing stages
            1. Data loaded into a `samples × timepoints × features` tensor.
            2. **Delta encoding** converts raw values into binary spike trains based on the threshold.
            3. **NeuCube reservoir** simulation learns spatio-temporal activity patterns via STDP.
            4. Spike counts are pooled into SNN feature vectors.
            5. A classifier is evaluated with stratified cross-validation.

            ### Main controls
            | Control | Effect |
            |---|---|
            | Spike sensitivity | How large a signal change generates a spike |
            | Reservoir C | Connectivity probability of the reservoir |
            | Reservoir L | Distance scaling for connectivity |
            | STDP parameters | Weight update rates during learning |
            | Membrane threshold | Neuron firing sensitivity |
            | Random seed | Fix for reproducible results |
        """)

    with tab_res:
        st.markdown("""
            ### How to read outputs
            | Metric | Meaning |
            |---|---|
            | Accuracy | Overall proportion of correct predictions |
            | Weighted F1 | Better metric for imbalanced classes |
            | Accuracy Lift | Improvement over a majority-class baseline |
            | Per-class metrics | Which classes are easy or hard to identify |
            | Confusion matrix | Which classes are confused with each other |

            Exported SNN features can be reused in notebooks or other modelling tools.
        """)

    with tab_about:
        st.markdown("""
            ### Credits
            Original SNN-QC toolbox and research prototype by **Dr. Ravi Kumar Jha**,
            Intelligent Systems Research Centre, Ulster University. Contact: **Jha-R@ulster.ac.uk**

            This repository extends the original work into a general-purpose Streamlit workbench
            for reusable time-series experiments.

            ### Data and materials
            The EEG dataset and NeuCube software environment are made available from Auckland
            University of Technology at https://kedri.aut.ac.nz/neucube.

            ### References
            1. Jha, R. K., Kasabov, N., Bhattacharyya, S., Coyle, D., & Prasad, G. (2025).
               A hybrid spiking neural network-quantum framework for spatio-temporal data classification.
               *EPJ Quantum Technology*, 12(1). https://doi.org/10.1140/epjqt/s40507-025-00443-1

            2. Kasabov, N. (2014). NeuCube: A spiking neural network architecture for mapping,
               learning and understanding of spatio-temporal brain data.
               *Neural Networks*, 52, 62–76. https://doi.org/10.1016/j.neunet.2014.01.006
        """)


# ── Workflow dashboard ─────────────────────────────────────────────────────────

def render_workflow_dashboard(threshold_val, res_c, res_l, mem_thr_val):
    data_ready = st.session_state.get("data_ready", False)
    map_ready = st.session_state.get("map_initialised", False)
    features_ready = st.session_state.get("features_ready", False)

    if not data_ready:
        next_step = "Load a built-in or uploaded dataset below."
    elif not map_ready:
        next_step = "Inspect encoded signals, then click Prepare Analysis Engine."
    elif not features_ready:
        next_step = "Click Run Feature Extraction to simulate the NeuCube reservoir."
    else:
        next_step = "Choose a model and click Generate Report."

    st.info(f"**Next step:** {next_step}")

    t_data, t_enc, t_sim, t_feat, t_rep = st.tabs(
        ["1 Data", "2 Prepare", "3 Analyze", "4 Features", "5 Report"]
    )

    with t_data:
        st.metric("Dataset", "Ready" if data_ready else "Not loaded")
        if data_ready:
            X = st.session_state["X"]
            y = st.session_state["y"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Samples", X.shape[0])
            c2.metric("Timepoints", X.shape[1])
            c3.metric("Features", X.shape[2])
            c4.metric("Classes", pd.Series(y).nunique())
            st.caption(f"Dataset: {st.session_state.get('dataset_name', '—')}")
        else:
            st.caption("Choose a data source below.")

    with t_enc:
        st.metric("Signal Preparation", "Encoded" if data_ready else "Waiting for data")
        st.caption(f"Delta threshold: {threshold_val}")
        if data_ready:
            X = st.session_state["X"]
            spike_density = float(np.count_nonzero(X.numpy()) / X.numel())
            st.metric("Spike Density", f"{spike_density:.2%}")

    with t_sim:
        status = "Features extracted" if features_ready else ("Engine ready" if map_ready else "Waiting")
        st.metric("Analysis Engine", status)
        c1, c2, c3 = st.columns(3)
        c1.metric("Reservoir C", res_c)
        c2.metric("Reservoir L", res_l)
        c3.metric("Membrane Threshold", mem_thr_val)

    with t_feat:
        st.metric("Spiking Features", "Ready" if features_ready else "Not available")
        if features_ready:
            st.write(f"Feature matrix shape: `{st.session_state['snn_features'].shape}`")
        else:
            st.caption("Run Feature Extraction to generate features.")

    with t_rep:
        st.metric("Model Report", "Ready to classify" if features_ready else "Waiting for features")
        st.caption("Classical SVM and Logistic Regression support multiclass. "
                   "Quantum Kernel SVM: binary, 2 features, experimental.")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

page = st.sidebar.radio("Navigation", ["Studio", "Help & Product Info"])
if page == "Help & Product Info":
    render_help_page()
    st.stop()

st.sidebar.header("Analysis Settings")
threshold_val = st.sidebar.slider(
    "Spike sensitivity", 0.1, 1.5, float(_defaults.get("spike_sensitivity", 0.8)), 0.05,
    help="How large a signal change must be to generate a spike. Lower = more spikes.",
)
k_folds = st.sidebar.slider(
    "Validation folds", 2, 10, int(_defaults.get("validation_folds", 5)),
    help="Number of stratified cross-validation folds.",
)
svm_c = st.sidebar.number_input(
    "Model flexibility (C)", value=float(_defaults.get("model_flexibility", 1.0)), min_value=0.001,
    help="SVM regularisation parameter. Higher = less regularisation.",
)

with st.sidebar.expander("Advanced engine settings"):
    res_c = st.slider(
        "Reservoir connectivity", 0.1, 1.0, float(_defaults.get("reservoir_c", 0.4)), 0.05,
        help="Probability of connection between reservoir neurons.",
    )
    res_l = st.slider(
        "Reservoir distance scale", 0.01, 1.0, float(_defaults.get("reservoir_l", 0.169)), 0.001,
        help="Distance decay for small-world connectivity.",
    )
    stdp_pos = st.number_input(
        "STDP positive update", value=float(_defaults.get("stdp_positive", 0.001)), format="%.4f",
        help="Weight increase when pre-synaptic spike precedes post-synaptic spike.",
    )
    stdp_neg = st.number_input(
        "STDP negative update", value=float(_defaults.get("stdp_negative", -0.01)), format="%.4f",
        help="Weight decrease for anti-causal spike pairs.",
    )
    mem_thr_val = st.number_input(
        "Neuron firing threshold", value=float(_defaults.get("membrane_threshold", 0.01)), format="%.3f",
        help="Membrane potential needed to fire a spike.",
    )

with st.sidebar.expander("Reproducibility"):
    seed_val = int(st.number_input(
        "Random seed", min_value=0, max_value=99999,
        value=int(_defaults.get("random_seed", 42)), step=1,
        help="Set a fixed seed for reproducible reservoir initialisation and cross-validation splits.",
    ))
    st.caption("Change to test result stability across different random initialisations.")

# Apply seed globally
np.random.seed(seed_val)
random.seed(seed_val)
torch.manual_seed(seed_val)

# ── State invalidation on param changes ───────────────────────────────────────

encoding_signature = {"delta_threshold": threshold_val}
simulation_signature = {
    "reservoir_c": res_c, "reservoir_l": res_l,
    "stdp_a_pos": stdp_pos, "stdp_a_neg": stdp_neg,
    "membrane_threshold": mem_thr_val,
}

if st.session_state.get("data_ready") and \
        st.session_state.get("encoding_signature") != encoding_signature:
    clear_dataset_state()
    st.session_state["workflow_notice"] = (
        "Spike sensitivity changed — dataset reset. Load and encode again."
    )
    st.rerun()

if st.session_state.get("features_ready") and \
        st.session_state.get("simulation_signature") != simulation_signature:
    clear_derived_state()
    st.session_state["workflow_notice"] = (
        "Engine parameters changed — extracted features reset. Re-run the simulation."
    )
    st.rerun()

if st.session_state.get("workflow_notice"):
    st.warning(st.session_state.pop("workflow_notice"))

# ── Sidebar footer + experiment history ───────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.info(
    f"**{APP_NAME}**\n\n{APP_TAGLINE}\n\n"
    "Original toolbox: **Dr. Ravi Kumar Jha**, Ulster University."
)

history = st.session_state.get("experiment_history", [])
if history:
    st.sidebar.markdown("**Experiment History**")
    st.sidebar.caption(f"{len(history)} run(s) this session.")
    history_df = pd.DataFrame(history)
    st.sidebar.download_button(
        "Download history (CSV)",
        dataframe_to_csv_bytes(history_df),
        file_name="experiment_history.csv",
        mime="text/csv",
    )
    if st.sidebar.button("Clear history"):
        st.session_state["experiment_history"] = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# WORKFLOW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

render_workflow_dashboard(threshold_val, res_c, res_l, mem_thr_val)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONNECT YOUR DATA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<h2 style='font-size:24px;'>Connect Your Data</h2>", unsafe_allow_html=True)

dataset_source = st.radio(
    "Choose a data source",
    ["Example dataset", "Single uploaded table", "Multiple sample files"],
    horizontal=True,
)

render_csv_templates()

combined_df = None
combined_config = {}
sample_files = []
sample_labels_df = None
sample_has_header = True

if dataset_source == "Example dataset":
    st.caption("Included EEG demo: 60 wrist-movement trials, 128 timepoints, 14 channels.")

elif dataset_source == "Single uploaded table":
    st.caption("One row per sample-timepoint. Columns: sample ID, label, numeric features.")
    combined_file = st.file_uploader("Upload combined CSV", type=["csv"], key="combined_csv")
    combined_has_header = st.checkbox("CSV has a header row", value=True)

    if combined_file is not None:
        combined_df = read_uploaded_csv(combined_file, combined_has_header)
        st.dataframe(combined_df.head(20), use_container_width=True)

        columns = [str(col) for col in combined_df.columns]
        combined_df.columns = columns

        c1, c2, c3 = st.columns(3)
        sample_col = c1.selectbox("Sample ID column", columns)
        label_col = c2.selectbox("Label column", columns, index=len(columns) - 1)
        time_col = c3.selectbox("Time/order column (optional)", ["(none)"] + columns)

        excluded = {sample_col, label_col} | ({time_col} if time_col != "(none)" else set())
        default_features = [col for col in columns if col not in excluded]
        feature_cols = st.multiselect("Feature columns", columns, default=default_features)

        combined_config = {
            "sample_col": sample_col, "label_col": label_col,
            "time_col": time_col, "feature_cols": feature_cols,
        }

        prev_X, prev_y, prev_feats, prev_err = parse_combined_csv(
            combined_df, sample_col, label_col, time_col, feature_cols
        )
        if prev_err:
            st.warning(prev_err)
        else:
            render_dataset_preview(prev_X, prev_y, prev_feats)

elif dataset_source == "Multiple sample files":
    st.caption(
        "Each sample CSV: timepoints × features. "
        "Labels CSV: one label per sample file, in sorted filename order."
    )
    sample_has_header = st.checkbox("Sample CSVs have a header row", value=True)
    sample_files = st.file_uploader(
        "Upload sample CSV files", type=["csv"], accept_multiple_files=True, key="sample_csvs"
    )
    sample_labels_file = st.file_uploader("Upload labels CSV", type=["csv"], key="sample_labels_csv")
    labels_has_header = st.checkbox("Labels CSV has a header row", value=False)

    if sample_files:
        sorted_names = sorted(f.name for f in sample_files)
        preview_names = ", ".join(sorted_names[:10]) + (" …" if len(sorted_names) > 10 else "")
        st.caption(f"Sample order for labels: {preview_names}")

    if sample_labels_file is not None:
        sample_labels_df = read_uploaded_csv(sample_labels_file, labels_has_header)
        st.dataframe(sample_labels_df.head(20), use_container_width=True)

    if sample_files and sample_labels_df is not None:
        prev_X, prev_y, prev_feats, prev_err = parse_sample_csvs(
            sample_files, sample_labels_df, sample_has_header
        )
        if prev_err:
            st.warning(prev_err)
        else:
            render_dataset_preview(prev_X, prev_y, prev_feats)


# ── Prepare Dataset button ─────────────────────────────────────────────────────

if st.button("Prepare Dataset", type="primary"):
    err = None
    with st.spinner("Loading and encoding dataset…"):
        if dataset_source == "Example dataset":
            X_raw, y_data, feature_names, err = load_builtin_eeg_dataset()
            dataset_name = "Example EEG dataset"

        elif dataset_source == "Single uploaded table":
            if combined_df is None:
                err = "Upload a combined CSV file first."
                X_raw = y_data = feature_names = None
            else:
                X_raw, y_data, feature_names, err = parse_combined_csv(
                    combined_df,
                    combined_config.get("sample_col"),
                    combined_config.get("label_col"),
                    combined_config.get("time_col"),
                    combined_config.get("feature_cols", []),
                )
            dataset_name = "Uploaded table"

        else:
            if not sample_files or sample_labels_df is None:
                err = "Upload both sample CSV files and a labels CSV file first."
                X_raw = y_data = feature_names = None
            else:
                X_raw, y_data, feature_names, err = parse_sample_csvs(
                    sample_files, sample_labels_df, sample_has_header
                )
            dataset_name = "Uploaded sample files"

    if err:
        st.error(err)
    else:
        with st.spinner("Applying delta encoding…"):
            X_encoded = encode_dataset(X_raw, threshold_val)

        clear_dataset_state()
        st.session_state.update({
            "X_raw": X_raw, "X": X_encoded, "y": y_data,
            "feature_names": feature_names, "dataset_name": dataset_name,
            "data_ready": True,
            "encoding_signature": encoding_signature,
            "simulation_signature": simulation_signature,
        })
        st.success("Dataset loaded and encoded successfully.")
        render_loaded_dataset_summary(X_encoded, y_data, feature_names)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATASET LOADED — SIGNAL PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("data_ready"):
    X_raw = st.session_state["X_raw"]
    X = st.session_state["X"]
    y = st.session_state["y"]
    feature_names = st.session_state.get(
        "feature_names", [f"Feature {i+1}" for i in range(X.shape[2])]
    )

    # Quantum SVM feature pickers (sidebar, only shown when data is ready)
    st.sidebar.header("Quantum SVM Features")
    st.sidebar.caption("Used only if Quantum Kernel SVM is selected as the model.")
    feat_name_1 = st.sidebar.selectbox("Primary feature", feature_names, index=0)
    feat_2_default = min(4, len(feature_names) - 1)
    feat_name_2 = st.sidebar.selectbox("Comparison feature", feature_names, index=feat_2_default)
    feat_idx_1 = feature_names.index(feat_name_1)
    feat_idx_2 = feature_names.index(feat_name_2)

    with st.expander("Dataset Summary", expanded=True):
        st.write(f"**Dataset:** {st.session_state.get('dataset_name', '—')}")
        render_loaded_dataset_summary(X, y, feature_names)

    with st.expander("Signal Preview", expanded=True):
        st.caption("Raw signal values vs. generated spike representation.")
        c1, c2 = st.columns(2)
        sel_sample = c1.slider("Select sample", 0, len(X) - 1, 0)
        sel_channel = c2.selectbox("Select feature", feature_names, index=0, key="viz_chan_sel")
        ch_idx = feature_names.index(sel_channel)
        fig = raw_and_spike_figure(
            X_raw[sel_sample, :, ch_idx].numpy(),
            X[sel_sample, :, ch_idx].numpy(),
            sel_channel, sel_sample,
        )
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: RUN ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown("<h2 style='font-size:24px;'>Run Analysis</h2>", unsafe_allow_html=True)

    if st.button("Prepare Analysis Engine"):
        st.session_state["simulation_signature"] = simulation_signature
        st.session_state["map_initialised"] = True

    if st.session_state.get("map_initialised"):
        with st.expander("Feature Map", expanded=True):
            c1, _ = st.columns([1, 4])
            clean_view = c1.toggle("Grid & Axes", value=True)
            st.plotly_chart(feature_layout_figure(feature_names, clean_view), use_container_width=True)

        st.divider()

        c1, c2 = st.columns([1, 3])
        run_sim = c1.button("Run Feature Extraction")

        if run_sim:
            gif_path = _HERE / "brain_activity.gif"
            with c2:
                brain_placeholder = st.empty()
                if gif_path.exists():
                    data_url = base64.b64encode(gif_path.read_bytes()).decode("utf-8")
                    brain_placeholder.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" '
                        f'width="300" style="border-radius:10px;">',
                        unsafe_allow_html=True,
                    )
                else:
                    brain_placeholder.info("Analysis running…")

            with st.spinner("Simulating NeuCube reservoir and extracting spiking features…"):
                res = Reservoir(inputs=X.shape[2], c=res_c, l=res_l)
                learning_rule = STDP(a_pos=stdp_pos, a_neg=stdp_neg)
                sam = SpikeCount()

                s_act_all = res.simulate(
                    X, train=True, learning_rule=learning_rule,
                    mem_thr=mem_thr_val, refractory_period=5, verbose=True,
                )
                state_vectors = sam.sample(s_act_all)
                snn_features = extract_features(state_vectors, res.w_in)

                st.session_state["snn_features"] = snn_features
                st.session_state["features_ready"] = True
                st.session_state["simulation_signature"] = simulation_signature

            brain_placeholder.empty()
            st.success("Feature extraction complete.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: FEATURE TABLE
    # ══════════════════════════════════════════════════════════════════════════

    if st.session_state.get("features_ready"):
        st.divider()
        st.markdown("<h2 style='font-size:24px;'>Feature Table</h2>", unsafe_allow_html=True)
        st.caption("Model-ready features extracted from the spiking neural network reservoir.")

        snn_features = st.session_state["snn_features"]
        st.write(f"**Feature matrix shape:** {snn_features.shape}")

        features_df = pd.DataFrame(snn_features, columns=feature_names)
        features_df.insert(0, "label", y)
        features_df.insert(0, "sample_index", np.arange(len(features_df)))
        st.download_button(
            "Download Feature Table (CSV)",
            dataframe_to_csv_bytes(features_df),
            file_name="snn_features.csv", mime="text/csv",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: MODEL REPORT
    # ══════════════════════════════════════════════════════════════════════════

    if st.session_state.get("features_ready"):
        snn_features = st.session_state["snn_features"]

        st.divider()
        st.markdown("<h2 style='font-size:24px;'>Model Report</h2>", unsafe_allow_html=True)

        available_classes = sorted(pd.Series(y).dropna().unique(), key=str)

        classifier_name = st.selectbox(
            "Model",
            [
                "Classical SVM",
                "Logistic Regression",
                "Quantum Kernel SVM  [experimental — binary, 2 features, slow]",
            ],
        )
        is_quantum = "Quantum" in classifier_name

        if is_quantum:
            st.warning(
                "**Quantum Kernel SVM** runs a classical simulation of a 2-qubit quantum circuit. "
                "It requires exactly **2 classes** and **2 features**, and can be very slow for "
                "large datasets. This is a research feature — use Classical SVM or Logistic "
                "Regression for reliable general-purpose results."
            )

        if len(available_classes) < 2:
            st.warning("At least two classes are required for classification.")
            st.stop()

        selected_classes = st.multiselect(
            "Classes for classification",
            available_classes,
            default=available_classes[:2] if is_quantum else available_classes,
        )

        if is_quantum and len(selected_classes) != 2:
            st.warning("Quantum Kernel SVM requires exactly two classes. Select exactly 2.")
            st.stop()
        if not is_quantum and len(selected_classes) < 2:
            st.warning("Choose at least two classes.")
            st.stop()

        class_mask = np.isin(y, selected_classes)
        filtered_features = snn_features[class_mask]
        y_final = y[class_mask]

        if is_quantum:
            feature_indices = [int(feat_idx_1), int(feat_idx_2)]
            selected_feature_names = [feat_name_1, feat_name_2]
        else:
            max_shown = int(_defaults.get("max_displayed_features", 14))
            selected_feature_names = st.multiselect(
                "Features for classification",
                feature_names,
                default=feature_names[:max_shown],
                help=f"First {max_shown} shown by default. Search or scroll to add more.",
            )
            if not selected_feature_names:
                st.warning("Choose at least one feature.")
                st.stop()
            feature_indices = [feature_names.index(n) for n in selected_feature_names]

        X_final = filtered_features[:, feature_indices]

        c1, c2 = st.columns(2)
        c1.write(f"**Classifier input shape:** `{X_final.shape}`")
        c2.write(f"**Selected features:** {', '.join(map(str, selected_feature_names))}")

        if is_quantum:
            with st.expander("Quantum Circuit"):
                try:
                    fig_qc, _ = qml.draw_mpl(kernel)(X_final[0], X_final[1])
                    st.pyplot(fig_qc)
                    plt.close(fig_qc)
                except Exception as exc:
                    st.info(f"Circuit diagram unavailable: {exc}")

        if st.button("Generate Report", type="primary"):
            class_counts = pd.Series(y_final).value_counts()
            effective_folds = min(k_folds, int(class_counts.min()))
            if effective_folds < 2:
                st.error(
                    "Each selected class needs at least 2 samples for cross-validation. "
                    "Add more samples or reduce the number of folds."
                )
                st.stop()

            kf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed_val)
            y_total, pred_total = [], []

            if is_quantum:
                classifier = SVC(kernel=kernel_matrix, C=svm_c)
            elif "Classical SVM" in classifier_name:
                classifier = SVC(kernel="rbf", C=svm_c, random_state=seed_val)
            else:
                classifier = LogisticRegression(max_iter=1000, random_state=seed_val)

            progress_bar = st.progress(0)
            status_txt = st.empty()

            for i, (train_idx, test_idx) in enumerate(kf.split(X_final, y_final)):
                status_txt.text(f"Cross-validation fold {i + 1} / {effective_folds}…")
                X_train, X_test = X_final[train_idx], X_final[test_idx]
                y_train, y_test = y_final[train_idx], y_final[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                classifier.fit(X_train, y_train)
                pred_total.extend(classifier.predict(X_test))
                y_total.extend(y_test)
                progress_bar.progress((i + 1) / effective_folds)

            status_txt.text("Validation complete.")

            final_acc = accuracy(y_total, pred_total)
            final_f1 = f1_score(y_total, pred_total, average="weighted", zero_division=0)

            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{final_acc:.2%}")
            c2.metric("Weighted F1", f"{final_f1:.2%}")

            interp = result_interpretation(final_acc, final_f1, y_total, pred_total)
            with st.expander("Result Interpretation", expanded=True):
                ci1, ci2 = st.columns(2)
                ci1.metric("Majority-Class Baseline", f"{interp['baseline']:.2%}")
                ci2.metric("Accuracy Lift", f"{interp['lift']:.2%}")
                st.write(interp["strength"])
                st.write(interp["confusion_note"])
                if len(y_total) < 30:
                    st.info("Small evaluation set — treat these metrics as an early signal, not a final claim.")

            report_df = pd.DataFrame(
                classification_report(y_total, pred_total, output_dict=True, zero_division=0)
            ).T
            st.markdown("##### Per-Class Metrics")
            st.dataframe(report_df, use_container_width=True)
            st.download_button(
                "Download Classification Metrics (CSV)",
                dataframe_to_csv_bytes(report_df.reset_index().rename(columns={"index": "class"})),
                file_name="classification_metrics.csv", mime="text/csv",
            )

            st.markdown("##### Confusion Matrix")
            cm = confusion_matrix(y_total, pred_total)
            unique_labels = sorted(set(y_total), key=str)
            c1, _ = st.columns([1, 2])
            with c1:
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels).plot(
                    cmap="Blues", ax=ax_cm, colorbar=False
                )
                ax_cm.set_title("Confusion Matrix", fontsize=7)
                ax_cm.set_xlabel("Predicted", fontsize=7)
                ax_cm.set_ylabel("True", fontsize=7)
                ax_cm.tick_params(labelsize=7)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            # Full experiment config + results as JSON
            exp_config = build_experiment_config(
                threshold_val, res_c, res_l, stdp_pos, stdp_neg, mem_thr_val, svm_c, k_folds, seed_val
            )
            exp_config["results"] = {"accuracy": round(final_acc, 4), "weighted_f1": round(final_f1, 4)}
            st.download_button(
                "Download Experiment Config (JSON)",
                json.dumps(exp_config, indent=2).encode("utf-8"),
                file_name="snnqc_experiment_config.json", mime="application/json",
            )

            # Save to in-session experiment history
            push_experiment_result(
                dataset_name=st.session_state.get("dataset_name", "—"),
                model_name=classifier_name.split("[")[0].strip(),
                n_samples=int(X_final.shape[0]),
                n_features=len(selected_feature_names),
                acc=final_acc,
                f1=final_f1,
                seed=seed_val,
            )
            st.success("Result saved to experiment history (downloadable from the sidebar).")
