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

# ── Constants ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

_config_path = _HERE / "config.yaml"
_cfg = yaml.safe_load(_config_path.read_text()) if _config_path.exists() else {}
_app_cfg = _cfg.get("app", {})
_defaults = _cfg.get("defaults", {})

APP_NAME    = _app_cfg.get("name",    "SpikeSense Studio")
APP_TAGLINE = _app_cfg.get("tagline", "Turn labelled time-series into spiking features and explainable model reports.")

DATASET_STATE_KEYS = [
    "X_raw", "X", "y", "feature_names", "dataset_name", "data_ready",
    "map_initialised", "features_ready", "snn_features",
]
DERIVED_STATE_KEYS = ["map_initialised", "features_ready", "snn_features"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="⚡")

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ───────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: #07070F;
    color: #E2E8F0;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 0 !important; max-width: 1200px; }
.main > div { padding-top: 1.5rem; }

/* ── Hero ───────────────────────────────────────────────────────────────── */
.ss-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0D0D1F 0%, #131028 50%, #0D1A1F 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 2.4rem 2.5rem;
    margin-bottom: 2rem;
}
.ss-hero::before {
    content: '';
    position: absolute; top: -80px; right: -60px;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(99,102,241,0.13) 0%, transparent 65%);
    pointer-events: none;
}
.ss-hero::after {
    content: '';
    position: absolute; bottom: -60px; left: 30%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(139,92,246,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.ss-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.07em; text-transform: uppercase;
    color: #818CF8; margin-bottom: 1.1rem;
}
.ss-hero h1 {
    font-size: 2.6rem; font-weight: 700; line-height: 1.1;
    margin: 0 0 0.8rem; color: #F8FAFC; letter-spacing: -0.03em;
}
.ss-hero h1 span { color: #818CF8; }
.ss-hero p { color: #94A3B8; font-size: 1.05rem; margin: 0; max-width: 580px; line-height: 1.65; }

/* ── Pipeline tracker ───────────────────────────────────────────────────── */
.ss-pipeline {
    display: flex; align-items: flex-start;
    background: #0C0C18;
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin: 1.5rem 0;
    gap: 0;
}
.ss-pipe-step {
    display: flex; flex-direction: column; align-items: center;
    gap: 8px; flex: 1; position: relative;
}
.ss-pipe-step:not(:last-child)::after {
    content: '';
    position: absolute; top: 16px; left: 55%; right: -45%;
    height: 2px; background: rgba(148,163,184,0.1);
}
.ss-pipe-step.done:not(:last-child)::after { background: linear-gradient(90deg, #6366F1, rgba(99,102,241,0.3)); }
.ss-pipe-dot {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; z-index: 1;
    flex-shrink: 0;
}
.ss-pipe-dot.done  { background: #6366F1; color: #fff; box-shadow: 0 0 12px rgba(99,102,241,0.5); }
.ss-pipe-dot.active{ background: rgba(99,102,241,0.15); color: #818CF8; border: 2px solid #6366F1; }
.ss-pipe-dot.pending{ background: rgba(148,163,184,0.06); color: #475569; border: 2px solid rgba(148,163,184,0.12); }
.ss-pipe-label { font-size: 0.68rem; font-weight: 500; text-align: center; letter-spacing: 0.03em; }
.ss-pipe-label.done    { color: #818CF8; }
.ss-pipe-label.active  { color: #A5B4FC; }
.ss-pipe-label.pending { color: #475569; }
.ss-next-step {
    display: flex; align-items: center; gap: 10px;
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 0.75rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem; color: #A5B4FC;
}
.ss-next-arrow { font-size: 1rem; }

/* ── Section headers ─────────────────────────────────────────────────────── */
.ss-section {
    display: flex; align-items: center; gap: 12px;
    margin: 2.2rem 0 1.2rem;
}
.ss-step-chip {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 8px;
    width: 30px; height: 30px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700; color: #fff;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(99,102,241,0.4);
}
.ss-section-title {
    font-size: 1.15rem; font-weight: 600; color: #F1F5F9; margin: 0;
}
.ss-section-sub { font-size: 0.85rem; color: #64748B; margin: 0; }

/* ── Cards ───────────────────────────────────────────────────────────────── */
.ss-card {
    background: #0F0F1C;
    border: 1px solid rgba(148,163,184,0.09);
    border-radius: 14px;
    padding: 1.5rem;
}
.ss-source-card {
    background: #0F0F1C;
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    cursor: pointer; transition: border-color 0.2s, background 0.2s;
}
.ss-source-card:hover {
    border-color: rgba(99,102,241,0.35);
    background: rgba(99,102,241,0.04);
}
.ss-source-card.active {
    border-color: rgba(99,102,241,0.5);
    background: rgba(99,102,241,0.08);
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0F0F1C, #12101E) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.3rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(99,102,241,0.3) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.08) !important;
}
[data-testid="stMetricLabel"] p {
    color: #64748B !important; font-size: 0.75rem !important;
    font-weight: 600 !important; text-transform: uppercase;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    color: #F1F5F9 !important; font-size: 1.7rem !important; font-weight: 700 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.18s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    border: none !important; color: #fff !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 20px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    color: #CBD5E1 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: rgba(99,102,241,0.4) !important; color: #A5B4FC !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] { border-bottom: 1px solid rgba(148,163,184,0.08); }
button[data-baseweb="tab"] {
    color: #64748B !important; font-weight: 500 !important;
    font-size: 0.88rem !important; padding: 0.6rem 1.1rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #818CF8 !important;
    border-bottom: 2px solid #6366F1 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #08080E !important;
    border-right: 1px solid rgba(99,102,241,0.1) !important;
}
section[data-testid="stSidebar"] * { color: #CBD5E1; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
section[data-testid="stSidebar"] .stSlider label { color: #94A3B8 !important; font-size: 0.85rem !important; }

/* ── Expanders ───────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: #0C0C18 !important;
    border: 1px solid rgba(148,163,184,0.09) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary { color: #94A3B8 !important; font-weight: 500; }
div[data-testid="stExpander"] summary:hover { color: #CBD5E1 !important; }

/* ── Dataframes ──────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148,163,184,0.09);
    border-radius: 10px; overflow: hidden;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
div[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Radio (data source) ─────────────────────────────────────────────────── */
div[data-testid="stRadio"] > label { color: #94A3B8 !important; font-size: 0.85rem !important; }
div[data-testid="stRadio"] [role="radiogroup"] { gap: 8px; }

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0C0C18 !important;
    border: 1px dashed rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: rgba(148,163,184,0.08) !important; margin: 2rem 0 !important; }

/* ── Toggle / checkbox ───────────────────────────────────────────────────── */
[data-testid="stToggle"] label { color: #94A3B8 !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #07070F; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_hero():
    st.markdown(f"""
    <div class="ss-hero">
        <div class="ss-badge">⚡ Signal Intelligence Platform</div>
        <h1>{APP_NAME}</h1>
        <p>{APP_TAGLINE}</p>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_tracker():
    data_ready     = st.session_state.get("data_ready",      False)
    map_ready      = st.session_state.get("map_initialised", False)
    features_ready = st.session_state.get("features_ready",  False)

    def _step(n, label, done, active):
        cls = "done" if done else ("active" if active else "pending")
        icon = "✓" if done else str(n)
        return f"""
        <div class="ss-pipe-step {cls}">
            <div class="ss-pipe-dot {cls}">{icon}</div>
            <div class="ss-pipe-label {cls}">{label}</div>
        </div>"""

    steps_html = (
        _step(1, "Load Data",   data_ready,                          not data_ready) +
        _step(2, "Encode",      data_ready,                          data_ready and not map_ready) +
        _step(3, "Analyse",     map_ready,                           data_ready and not map_ready) +
        _step(4, "Features",    features_ready,                      map_ready and not features_ready) +
        _step(5, "Report",      False,                               features_ready)
    )

    st.markdown(f'<div class="ss-pipeline">{steps_html}</div>', unsafe_allow_html=True)

    if not data_ready:
        msg = "Load a dataset below to begin."
    elif not map_ready:
        msg = "Inspect your encoded signals, then click <b>Prepare Analysis Engine</b>."
    elif not features_ready:
        msg = "Click <b>Run Feature Extraction</b> to simulate the reservoir."
    else:
        msg = "Choose a model and click <b>Generate Report</b>."

    st.markdown(
        f'<div class="ss-next-step"><span class="ss-next-arrow">→</span>{msg}</div>',
        unsafe_allow_html=True,
    )


def section_header(step: int, title: str, subtitle: str = ""):
    sub_html = f'<p class="ss-section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="ss-section">
        <div class="ss-step-chip">{step}</div>
        <div>
            <p class="ss-section-title">{title}</p>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def status_pill(label: str, ready: bool) -> str:
    if ready:
        return f'<span style="background:rgba(16,185,129,.12);color:#34D399;border:1px solid rgba(16,185,129,.2);border-radius:100px;padding:2px 10px;font-size:.72rem;font-weight:600;">{label}</span>'
    return f'<span style="background:rgba(100,116,139,.1);color:#64748B;border:1px solid rgba(100,116,139,.18);border-radius:100px;padding:2px 10px;font-size:.72rem;font-weight:600;">{label}</span>'


# ── State helpers ─────────────────────────────────────────────────────────────

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


def encode_dataset(X_raw, thresh):
    encoder = Delta(threshold=thresh)
    return encoder.encode_dataset(X_raw)


# ── Data render helpers ───────────────────────────────────────────────────────

def render_csv_templates():
    with st.expander("Download CSV Templates"):
        c1, c2, c3 = st.columns(3)
        c1.download_button("Combined table template",  dataframe_to_csv_bytes(combined_csv_template()),
                           file_name="snnqc_combined_template.csv",  mime="text/csv")
        c2.download_button("Sample file template",     dataframe_to_csv_bytes(sample_csv_template()),
                           file_name="snnqc_sample_template.csv",    mime="text/csv")
        c3.download_button("Labels file template",     dataframe_to_csv_bytes(labels_csv_template()),
                           file_name="snnqc_labels_template.csv",    mime="text/csv")


def render_dataset_preview(X_raw, y_data, feature_names):
    preview = dataset_summary(X_raw, y_data, feature_names)
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples",    preview["samples"])
    c2.metric("Timepoints", preview["timepoints"])
    c3.metric("Features",   preview["features"])
    st.caption("Class distribution")
    st.dataframe(
        preview["class_counts"].rename("count").reset_index().rename(columns={"index": "class"}),
        use_container_width=True,
    )


def render_loaded_dataset_summary(X, y, feature_names):
    summary = dataset_summary(X, y, feature_names)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples",    summary["samples"])
    c2.metric("Timepoints", summary["timepoints"])
    c3.metric("Features",   summary["features"])
    c4.metric("Classes",    len(summary["class_counts"]))
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
        strength = "Slightly above baseline — treat this as an early signal."
    else:
        strength = "Not yet beating the majority-class baseline."

    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true), key=str)
    confusion_note = "No off-diagonal class confusion observed."
    if cm.shape[0] > 1:
        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)
        if off_diag.max() > 0:
            r, c = np.unravel_index(off_diag.argmax(), off_diag.shape)
            confusion_note = (
                f"Most common confusion: true '{labels[r]}' predicted as "
                f"'{labels[c]}' ({off_diag[r, c]} samples)."
            )

    return {"baseline": baseline, "lift": lift, "strength": strength, "confusion_note": confusion_note}


def push_experiment_result(dataset_name, model_name, n_samples, n_features, acc, f1, seed):
    if "experiment_history" not in st.session_state:
        st.session_state["experiment_history"] = []
    st.session_state["experiment_history"].append({
        "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":      dataset_name,
        "model":        model_name,
        "samples":      n_samples,
        "features_used": n_features,
        "accuracy":     round(float(acc), 4),
        "weighted_f1":  round(float(f1), 4),
        "seed":         seed,
    })


def build_experiment_config(threshold_val, res_c, res_l, stdp_pos, stdp_neg,
                             mem_thr_val, svm_c, k_folds, seed):
    config = {
        "app": APP_NAME,
        "dataset": st.session_state.get("dataset_name"),
        "hyperparameters": {
            "delta_threshold": threshold_val, "reservoir_c": res_c,
            "reservoir_l": res_l, "stdp_a_pos": stdp_pos, "stdp_a_neg": stdp_neg,
            "membrane_threshold": mem_thr_val, "svm_c": svm_c,
            "cv_folds": k_folds, "random_seed": seed,
        },
    }
    if st.session_state.get("data_ready"):
        X, y = st.session_state["X"], st.session_state["y"]
        config["data"] = {
            "shape": list(X.shape),
            "classes": [str(v) for v in sorted(pd.Series(y).dropna().unique(), key=str)],
            "feature_names": [str(v) for v in st.session_state.get("feature_names", [])],
        }
    if st.session_state.get("features_ready"):
        config["snn_features"] = {"shape": list(st.session_state["snn_features"].shape)}
    return config


# ── Help page ─────────────────────────────────────────────────────────────────

def render_help_page():
    render_hero()
    st.markdown("## Documentation")

    tab_ov, tab_data, tab_pipe, tab_res, tab_about = st.tabs(
        ["Overview", "Data Formats", "Pipeline", "Interpreting Results", "About"]
    )

    with tab_ov:
        st.markdown("""
            ### What SpikeSense Studio does
            Upload any labelled time-series dataset and get a full classification report in minutes:
            spike-encoded features, cross-validated accuracy, per-class metrics, and a downloadable
            experiment record.

            **Works with:** EEG, motion sensors, wearables, industrial monitoring, clinical signals,
            financial time-series — any fixed-length repeated measurements with class labels.

            ### Requirements
            - Every sample must have the **same number of timepoints**.
            - At least **2 labelled classes** and **2 samples per class**.
            - Numeric feature values (no text, no missing values).

            ### Model options
            | Model | Best for |
            |---|---|
            | **Classical SVM** | General purpose, multiclass, robust baseline |
            | **Logistic Regression** | Interpretable, fast, multiclass |
            | **Quantum Kernel SVM** | Research only — binary, 2 features, slow |
        """)

    with tab_data:
        st.markdown("""
            ### Option A — Single combined table
            One row per sample-timepoint. Select columns in the UI after upload.

            | sample_id | time | label | sensor_1 | sensor_2 |
            |---|---|---|---|---|
            | trial_001 | 0 | class_a | 0.12 | 0.32 |
            | trial_001 | 1 | class_a | 0.18 | 0.29 |
            | trial_002 | 0 | class_b | 0.76 | 0.15 |

            ### Option B — Multiple sample files
            One CSV per sample (rows = timepoints, columns = features).
            Upload a separate labels CSV with **one label per file**, matched in sorted filename order.

            Download the templates from the **Download CSV Templates** expander on the main page.
        """)

    with tab_pipe:
        st.markdown("""
            ### How SpikeSense Studio processes your data

            ```
            Raw signal  →  Delta encoding  →  NeuCube reservoir  →  Feature extraction  →  Classifier
            (continuous)    (spike trains)     (STDP learning)       (spike counts)         (SVM / LR)
            ```

            | Stage | What happens |
            |---|---|
            | **Delta encoding** | Changes larger than the threshold become spikes (1), others stay 0 |
            | **NeuCube reservoir** | A 10×10×10 spiking neural network learns temporal patterns via STDP |
            | **Feature extraction** | Spike counts per neuron are pooled back to input channels |
            | **Classification** | Stratified k-fold cross-validation with your chosen model |

            ### Controls guide
            | Parameter | Effect |
            |---|---|
            | Spike sensitivity | Lower → more spikes, higher → fewer spikes |
            | Reservoir C | Connection probability between reservoir neurons |
            | Reservoir L | Distance decay — higher = longer-range connections |
            | Membrane threshold | How much charge a neuron needs to fire |
            | Random seed | Pin for reproducible results |
        """)

    with tab_res:
        st.markdown("""
            ### Reading your results

            | Metric | Meaning |
            |---|---|
            | **Accuracy** | % of test samples correctly classified |
            | **Weighted F1** | Accuracy adjusted for class imbalance — prefer this for uneven datasets |
            | **Accuracy Lift** | How much better than a naive majority-class classifier |
            | **Per-class precision** | When the model predicts this class, how often is it right? |
            | **Per-class recall** | Of all actual samples in this class, how many did the model catch? |
            | **Confusion matrix** | Which classes get confused with each other |

            ### What counts as a good result?
            - **Lift > 15%** above baseline = the spiking features are carrying signal.
            - **Lift 5–15%** = exploratory — tune the threshold and reservoir parameters.
            - **Lift < 5%** = the encoding or feature extraction may need adjustment.
        """)

    with tab_about:
        st.markdown("""
            ### Credits
            Original SNN-QC toolbox and research prototype by **Dr. Ravi Kumar Jha**,
            Intelligent Systems Research Centre, Ulster University. Contact: **Jha-R@ulster.ac.uk**

            This repository extends the original work into a general-purpose analysis product.

            ### References
            1. Jha et al. (2025). A hybrid SNN-quantum framework for spatio-temporal data classification.
               *EPJ Quantum Technology*, 12(1). https://doi.org/10.1140/epjqt/s40507-025-00443-1

            2. Kasabov, N. (2014). NeuCube: A spiking neural network architecture for mapping,
               learning and understanding of spatio-temporal brain data.
               *Neural Networks*, 52, 62–76. https://doi.org/10.1016/j.neunet.2014.01.006

            ### Data
            EEG dataset and NeuCube environment from Auckland University of Technology —
            https://kedri.aut.ac.nz/neucube
        """)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

page = st.sidebar.radio("", ["Studio", "Documentation"], label_visibility="collapsed")
if page == "Documentation":
    render_help_page()
    st.stop()

# Sidebar branding
st.sidebar.markdown("""
<div style="padding:1.2rem 0.5rem 0.8rem;border-bottom:1px solid rgba(99,102,241,0.12);margin-bottom:1rem;">
    <div style="font-size:1.1rem;font-weight:700;color:#F1F5F9;">⚡ SpikeSense Studio</div>
    <div style="font-size:0.75rem;color:#475569;margin-top:3px;">Signal intelligence platform</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**Analysis Settings**")
threshold_val = st.sidebar.slider(
    "Spike sensitivity", 0.1, 1.5, float(_defaults.get("spike_sensitivity", 0.8)), 0.05,
    help="Minimum signal change required to generate a spike. Lower = more spikes.",
)
k_folds = st.sidebar.slider(
    "Validation folds", 2, 10, int(_defaults.get("validation_folds", 5)),
    help="Number of stratified cross-validation folds.",
)
svm_c = st.sidebar.number_input(
    "Model flexibility (C)", value=float(_defaults.get("model_flexibility", 1.0)),
    min_value=0.001,
    help="SVM regularisation. Higher = less regularisation.",
)

with st.sidebar.expander("Advanced engine settings"):
    res_c = st.slider(
        "Reservoir connectivity", 0.1, 1.0, float(_defaults.get("reservoir_c", 0.4)), 0.05,
    )
    res_l = st.slider(
        "Distance scale", 0.01, 1.0, float(_defaults.get("reservoir_l", 0.169)), 0.001,
    )
    stdp_pos = st.number_input(
        "STDP positive update", value=float(_defaults.get("stdp_positive", 0.001)), format="%.4f",
    )
    stdp_neg = st.number_input(
        "STDP negative update", value=float(_defaults.get("stdp_negative", -0.01)), format="%.4f",
    )
    mem_thr_val = st.number_input(
        "Membrane threshold", value=float(_defaults.get("membrane_threshold", 0.01)), format="%.3f",
    )
    reservoir_size = st.select_slider(
        "Reservoir size",
        options=["Fast (5×5×5)", "Standard (7×7×7)", "Full (10×10×10)"],
        value=_defaults.get("reservoir_size", "Standard (7×7×7)"),
        help="Larger = better features but slower. Fast ≈30s, Standard ≈90s, Full ≈3–5min on cloud.",
    )

_CUBE_SHAPES = {
    "Fast (5×5×5)":    (5, 5, 5),
    "Standard (7×7×7)":(7, 7, 7),
    "Full (10×10×10)": (10,10,10),
}
_CUBE_TIMES = {
    "Fast (5×5×5)":    "~30 seconds",
    "Standard (7×7×7)":"~90 seconds",
    "Full (10×10×10)": "~3–5 minutes",
}
cube_shape = _CUBE_SHAPES[reservoir_size]

with st.sidebar.expander("Reproducibility"):
    seed_val = int(st.number_input(
        "Random seed", min_value=0, max_value=99999,
        value=int(_defaults.get("random_seed", 42)), step=1,
        help="Fix for reproducible reservoir initialisation and CV splits.",
    ))

np.random.seed(seed_val)
random.seed(seed_val)
torch.manual_seed(seed_val)

# ── State invalidation ────────────────────────────────────────────────────────

encoding_signature   = {"delta_threshold": threshold_val}
simulation_signature = {
    "reservoir_c": res_c, "reservoir_l": res_l,
    "stdp_a_pos": stdp_pos, "stdp_a_neg": stdp_neg,
    "membrane_threshold": mem_thr_val, "cube_shape": cube_shape,
}

if st.session_state.get("data_ready") and \
        st.session_state.get("encoding_signature") != encoding_signature:
    clear_dataset_state()
    st.session_state["workflow_notice"] = "Spike sensitivity changed — dataset reset. Reload to continue."
    st.rerun()

if st.session_state.get("features_ready") and \
        st.session_state.get("simulation_signature") != simulation_signature:
    clear_derived_state()
    st.session_state["workflow_notice"] = "Engine parameters changed — features reset. Re-run the simulation."
    st.rerun()

if st.session_state.get("workflow_notice"):
    st.warning(st.session_state.pop("workflow_notice"))

# ── Sidebar: experiment history ───────────────────────────────────────────────

history = st.session_state.get("experiment_history", [])
if history:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Experiment History** — {len(history)} run(s)")
    history_df = pd.DataFrame(history)
    st.sidebar.download_button(
        "Download history (CSV)",
        dataframe_to_csv_bytes(history_df),
        file_name="experiment_history.csv", mime="text/csv",
    )
    if st.sidebar.button("Clear history"):
        st.session_state["experiment_history"] = []
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Original research: Dr. Ravi Kumar Jha, Ulster University")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — HERO + PIPELINE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

render_hero()
render_pipeline_tracker()


# ══════════════════════════════════════════════════════════════════════════════
# WORKFLOW TABS
# ══════════════════════════════════════════════════════════════════════════════

t_data, t_enc, t_sim, t_feat, t_rep = st.tabs(
    ["  Data  ", "  Prepare  ", "  Analyse  ", "  Features  ", "  Report  "]
)

data_ready     = st.session_state.get("data_ready",      False)
map_ready      = st.session_state.get("map_initialised", False)
features_ready = st.session_state.get("features_ready",  False)

with t_data:
    st.markdown(
        status_pill("Ready", data_ready) if data_ready else status_pill("Not loaded", False),
        unsafe_allow_html=True,
    )
    st.write("")
    if data_ready:
        X = st.session_state["X"]
        y = st.session_state["y"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples",    X.shape[0])
        c2.metric("Timepoints", X.shape[1])
        c3.metric("Features",   X.shape[2])
        c4.metric("Classes",    pd.Series(y).nunique())
        st.caption(f"Dataset: {st.session_state.get('dataset_name', '—')}")
    else:
        st.caption("Load a dataset below to begin.")

with t_enc:
    st.markdown(status_pill("Encoded", data_ready) if data_ready else status_pill("Waiting", False),
                unsafe_allow_html=True)
    st.write("")
    st.caption(f"Spike sensitivity threshold: **{threshold_val}**")
    if data_ready:
        X = st.session_state["X"]
        spike_density = float(np.count_nonzero(X.numpy()) / X.numel())
        st.metric("Spike Density", f"{spike_density:.2%}")

with t_sim:
    label = "Features extracted" if features_ready else ("Engine ready" if map_ready else "Waiting")
    st.markdown(status_pill(label, features_ready or map_ready), unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Connectivity C",     res_c)
    c2.metric("Distance Scale L",   res_l)
    c3.metric("Membrane Threshold", mem_thr_val)

with t_feat:
    st.markdown(status_pill("Ready", features_ready) if features_ready else status_pill("Waiting", False),
                unsafe_allow_html=True)
    st.write("")
    if features_ready:
        st.write(f"Feature matrix: `{st.session_state['snn_features'].shape}`")
    else:
        st.caption("Run Feature Extraction to generate the spiking feature matrix.")

with t_rep:
    st.markdown(status_pill("Ready", features_ready) if features_ready else status_pill("Waiting", False),
                unsafe_allow_html=True)
    st.write("")
    st.caption("Classical SVM and Logistic Regression support multiclass. Quantum Kernel SVM: binary, 2 features, experimental.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CONNECT YOUR DATA
# ══════════════════════════════════════════════════════════════════════════════

section_header(1, "Connect Your Data", "Choose a source and load your labelled time-series dataset.")

dataset_source = st.radio(
    "Data source",
    ["Example dataset", "Single uploaded table", "Multiple sample files"],
    horizontal=True,
    label_visibility="collapsed",
)

render_csv_templates()

combined_df     = None
combined_config = {}
sample_files    = []
sample_labels_df = None
sample_has_header = True

if dataset_source == "Example dataset":
    st.info("**EEG demo dataset** — 60 wrist-movement trials, 128 timepoints, 14 EEG channels. Ready to go.")

elif dataset_source == "Single uploaded table":
    st.caption("One row per sample-timepoint. Columns: sample ID, label, numeric features.")
    combined_file     = st.file_uploader("Upload combined CSV", type=["csv"], key="combined_csv")
    combined_has_header = st.checkbox("File has a header row", value=True)

    if combined_file is not None:
        combined_df = read_uploaded_csv(combined_file, combined_has_header)
        st.dataframe(combined_df.head(10), use_container_width=True)

        columns = [str(col) for col in combined_df.columns]
        combined_df.columns = columns

        c1, c2, c3 = st.columns(3)
        sample_col = c1.selectbox("Sample ID column", columns)
        label_col  = c2.selectbox("Label column",     columns, index=len(columns) - 1)
        time_col   = c3.selectbox("Time column (optional)", ["(none)"] + columns)

        excluded       = {sample_col, label_col} | ({time_col} if time_col != "(none)" else set())
        default_feats  = [col for col in columns if col not in excluded]
        feature_cols   = st.multiselect("Feature columns", columns, default=default_feats)

        combined_config = {
            "sample_col": sample_col, "label_col": label_col,
            "time_col": time_col,     "feature_cols": feature_cols,
        }

        prev_X, prev_y, prev_feats, prev_err = parse_combined_csv(
            combined_df, sample_col, label_col, time_col, feature_cols
        )
        if prev_err:
            st.warning(prev_err)
        else:
            render_dataset_preview(prev_X, prev_y, prev_feats)

elif dataset_source == "Multiple sample files":
    st.caption("One CSV per sample (timepoints × features). Labels CSV: one label per file, sorted filename order.")
    sample_has_header = st.checkbox("Sample CSVs have a header row", value=True)
    sample_files = st.file_uploader(
        "Upload sample CSV files", type=["csv"], accept_multiple_files=True, key="sample_csvs"
    )
    sample_labels_file = st.file_uploader("Upload labels CSV", type=["csv"], key="sample_labels_csv")
    labels_has_header  = st.checkbox("Labels CSV has a header row", value=False)

    if sample_files:
        sorted_names = sorted(f.name for f in sample_files)
        st.caption("File order: " + ", ".join(sorted_names[:8]) + (" …" if len(sorted_names) > 8 else ""))

    if sample_labels_file is not None:
        sample_labels_df = read_uploaded_csv(sample_labels_file, labels_has_header)
        st.dataframe(sample_labels_df.head(10), use_container_width=True)

    if sample_files and sample_labels_df is not None:
        prev_X, prev_y, prev_feats, prev_err = parse_sample_csvs(
            sample_files, sample_labels_df, sample_has_header
        )
        if prev_err:
            st.warning(prev_err)
        else:
            render_dataset_preview(prev_X, prev_y, prev_feats)

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
                err = "Upload both sample CSV files and a labels CSV first."
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
        st.success(f"Dataset ready — {X_raw.shape[0]} samples, {X_raw.shape[1]} timepoints, {X_raw.shape[2]} features.")
        render_loaded_dataset_summary(X_encoded, y_data, feature_names)


# ══════════════════════════════════════════════════════════════════════════════
# LOADED — SIGNAL PREVIEW + ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("data_ready"):
    X_raw        = st.session_state["X_raw"]
    X            = st.session_state["X"]
    y            = st.session_state["y"]
    feature_names = st.session_state.get(
        "feature_names", [f"Feature {i+1}" for i in range(X.shape[2])]
    )

    # Quantum SVM feature pickers (sidebar)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quantum SVM Features**")
    st.sidebar.caption("Only used if Quantum Kernel SVM is selected.")
    feat_name_1 = st.sidebar.selectbox("Primary feature",    feature_names, index=0)
    feat_2_default = min(4, len(feature_names) - 1)
    feat_name_2 = st.sidebar.selectbox("Comparison feature", feature_names, index=feat_2_default)
    feat_idx_1 = feature_names.index(feat_name_1)
    feat_idx_2 = feature_names.index(feat_name_2)

    with st.expander("Dataset Summary", expanded=True):
        st.caption(f"**{st.session_state.get('dataset_name', '—')}**")
        render_loaded_dataset_summary(X, y, feature_names)

    # ── STEP 2: Signal Preview ────────────────────────────────────────────────
    section_header(2, "Signal Preview", "Inspect raw signal values and the generated spike representation.")

    with st.container():
        c1, c2 = st.columns(2)
        sel_sample  = c1.slider("Sample", 0, len(X) - 1, 0)
        sel_channel = c2.selectbox("Feature", feature_names, index=0, key="viz_chan_sel")
        ch_idx = feature_names.index(sel_channel)
        fig = raw_and_spike_figure(
            X_raw[sel_sample, :, ch_idx].numpy(),
            X[sel_sample, :, ch_idx].numpy(),
            sel_channel, sel_sample,
        )
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ── STEP 3: Analysis Engine ───────────────────────────────────────────────
    section_header(3, "Run Analysis", "Initialise the reservoir, then extract spiking features.")

    if st.button("Prepare Analysis Engine"):
        st.session_state["simulation_signature"] = simulation_signature
        st.session_state["map_initialised"] = True

    if st.session_state.get("map_initialised"):
        with st.expander("Feature Map", expanded=True):
            c1, _ = st.columns([1, 4])
            clean_view = c1.toggle("Clean view", value=True)
            st.plotly_chart(feature_layout_figure(feature_names, clean_view), use_container_width=True)

        st.divider()

        n_neurons = cube_shape[0] * cube_shape[1] * cube_shape[2]
        st.caption(
            f"Reservoir: **{reservoir_size}** — {n_neurons} neurons — "
            f"estimated time on cloud: **{_CUBE_TIMES[reservoir_size]}**"
        )

        c1, c2 = st.columns([1, 3])
        run_sim = c1.button("Run Feature Extraction", type="primary")

        if run_sim:
            gif_path = _HERE / "brain_activity.gif"
            with c2:
                brain_ph = st.empty()
                if gif_path.exists():
                    data_url = base64.b64encode(gif_path.read_bytes()).decode("utf-8")
                    brain_ph.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" '
                        f'width="280" style="border-radius:12px;opacity:0.85;">',
                        unsafe_allow_html=True,
                    )
                else:
                    brain_ph.info("Simulation running…")

            with st.spinner(f"Simulating reservoir ({n_neurons} neurons × {X.shape[0]} samples)… {_CUBE_TIMES[reservoir_size]}"):
                res           = Reservoir(cube_shape=cube_shape, inputs=X.shape[2], c=res_c, l=res_l)
                learning_rule = STDP(a_pos=stdp_pos, a_neg=stdp_neg)
                sam           = SpikeCount()

                s_act_all    = res.simulate(
                    X, train=True, learning_rule=learning_rule,
                    mem_thr=mem_thr_val, refractory_period=5, verbose=True,
                )
                state_vectors = sam.sample(s_act_all)
                snn_features  = extract_features(state_vectors, res.w_in)

                st.session_state["snn_features"]        = snn_features
                st.session_state["features_ready"]       = True
                st.session_state["simulation_signature"] = simulation_signature

            brain_ph.empty()
            st.success("Feature extraction complete.")

    # ── STEP 4: Feature Table ─────────────────────────────────────────────────
    if st.session_state.get("features_ready"):
        st.divider()
        section_header(4, "Feature Table", "Download the model-ready spiking feature matrix.")

        snn_features = st.session_state["snn_features"]
        c1, c2 = st.columns([3, 1])
        c1.write(f"Shape: `{snn_features.shape}` — {snn_features.shape[0]} samples × {snn_features.shape[1]} features")

        features_df = pd.DataFrame(snn_features, columns=feature_names)
        features_df.insert(0, "label",        y)
        features_df.insert(0, "sample_index", np.arange(len(features_df)))
        c2.download_button(
            "Download features (CSV)",
            dataframe_to_csv_bytes(features_df),
            file_name="snn_features.csv", mime="text/csv",
        )

    # ── STEP 5: Model Report ──────────────────────────────────────────────────
    if st.session_state.get("features_ready"):
        snn_features = st.session_state["snn_features"]

        st.divider()
        section_header(5, "Model Report", "Choose a classifier, run cross-validation, and download the results.")

        available_classes = sorted(pd.Series(y).dropna().unique(), key=str)

        classifier_name = st.selectbox(
            "Classifier",
            [
                "Classical SVM",
                "Logistic Regression",
                "Quantum Kernel SVM  [experimental — binary, 2 features, slow]",
            ],
        )
        is_quantum = "Quantum" in classifier_name

        if is_quantum:
            st.warning(
                "**Quantum Kernel SVM** simulates a 2-qubit quantum circuit classically. "
                "Requires exactly **2 classes** and **2 features**. "
                "Can be very slow for large datasets. For general use, choose Classical SVM or Logistic Regression."
            )

        if len(available_classes) < 2:
            st.warning("At least two classes are required.")
            st.stop()

        selected_classes = st.multiselect(
            "Classes to include",
            available_classes,
            default=available_classes[:2] if is_quantum else available_classes,
        )
        if is_quantum and len(selected_classes) != 2:
            st.warning("Select exactly 2 classes for the Quantum Kernel SVM.")
            st.stop()
        if not is_quantum and len(selected_classes) < 2:
            st.warning("Select at least 2 classes.")
            st.stop()

        class_mask       = np.isin(y, selected_classes)
        filtered_features = snn_features[class_mask]
        y_final          = y[class_mask]

        if is_quantum:
            feature_indices      = [int(feat_idx_1), int(feat_idx_2)]
            selected_feature_names = [feat_name_1, feat_name_2]
        else:
            max_shown = int(_defaults.get("max_displayed_features", 14))
            selected_feature_names = st.multiselect(
                "Features to use",
                feature_names,
                default=feature_names[:max_shown],
                help=f"First {max_shown} shown by default. Search to add more.",
            )
            if not selected_feature_names:
                st.warning("Select at least one feature.")
                st.stop()
            feature_indices = [feature_names.index(n) for n in selected_feature_names]

        X_final = filtered_features[:, feature_indices]

        c1, c2 = st.columns(2)
        c1.caption(f"Input shape: `{X_final.shape}`")
        c2.caption(f"Features: {', '.join(map(str, selected_feature_names))}")

        if is_quantum:
            with st.expander("Quantum Circuit"):
                try:
                    fig_qc, _ = qml.draw_mpl(kernel)(X_final[0], X_final[1])
                    st.pyplot(fig_qc)
                    plt.close(fig_qc)
                except Exception as exc:
                    st.info(f"Circuit diagram unavailable: {exc}")

        if st.button("Generate Report", type="primary"):
            class_counts    = pd.Series(y_final).value_counts()
            effective_folds = min(k_folds, int(class_counts.min()))
            if effective_folds < 2:
                st.error("Each class needs at least 2 samples for cross-validation.")
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
            status_txt   = st.empty()

            for i, (train_idx, test_idx) in enumerate(kf.split(X_final, y_final)):
                status_txt.caption(f"Cross-validation fold {i+1} / {effective_folds}…")
                X_train, X_test = X_final[train_idx], X_final[test_idx]
                y_train, y_test = y_final[train_idx], y_final[test_idx]

                scaler  = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

                classifier.fit(X_train, y_train)
                pred_total.extend(classifier.predict(X_test))
                y_total.extend(y_test)
                progress_bar.progress((i + 1) / effective_folds)

            status_txt.empty()
            progress_bar.empty()

            final_acc = accuracy(y_total, pred_total)
            final_f1  = f1_score(y_total, pred_total, average="weighted", zero_division=0)

            # ── Results ───────────────────────────────────────────────────────
            st.markdown("##### Results")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy",    f"{final_acc:.2%}")
            c2.metric("Weighted F1", f"{final_f1:.2%}")

            interp = result_interpretation(final_acc, final_f1, y_total, pred_total)
            with st.expander("Result Interpretation", expanded=True):
                ci1, ci2 = st.columns(2)
                ci1.metric("Majority-Class Baseline", f"{interp['baseline']:.2%}")
                ci2.metric("Accuracy Lift",           f"{interp['lift']:.2%}")
                st.write(interp["strength"])
                st.write(interp["confusion_note"])
                if len(y_total) < 30:
                    st.info("Small evaluation set — use these metrics as an early signal only.")

            # ── Per-class metrics ─────────────────────────────────────────────
            st.markdown("##### Per-Class Metrics")
            report_df = pd.DataFrame(
                classification_report(y_total, pred_total, output_dict=True, zero_division=0)
            ).T
            st.dataframe(report_df, use_container_width=True)

            # ── Confusion matrix ──────────────────────────────────────────────
            st.markdown("##### Confusion Matrix")
            cm            = confusion_matrix(y_total, pred_total)
            unique_labels = sorted(set(y_total), key=str)
            c1, _ = st.columns([1, 2])
            with c1:
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3),
                                             facecolor="#0C0C18")
                ax_cm.set_facecolor("#0C0C18")
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels).plot(
                    cmap="Blues", ax=ax_cm, colorbar=False,
                )
                ax_cm.set_title("Confusion Matrix", fontsize=7, color="#94A3B8")
                ax_cm.set_xlabel("Predicted", fontsize=7, color="#64748B")
                ax_cm.set_ylabel("True",      fontsize=7, color="#64748B")
                ax_cm.tick_params(labelsize=7, colors="#94A3B8")
                for spine in ax_cm.spines.values():
                    spine.set_edgecolor((148/255, 163/255, 184/255, 0.15))
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            # ── Downloads ────────────────────────────────────────────────────
            st.markdown("##### Export")
            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "Download metrics (CSV)",
                dataframe_to_csv_bytes(report_df.reset_index().rename(columns={"index": "class"})),
                file_name="classification_metrics.csv", mime="text/csv",
            )
            exp_config = build_experiment_config(
                threshold_val, res_c, res_l, stdp_pos, stdp_neg, mem_thr_val, svm_c, k_folds, seed_val
            )
            exp_config["results"] = {"accuracy": round(final_acc, 4), "weighted_f1": round(final_f1, 4)}
            dl2.download_button(
                "Download experiment config (JSON)",
                json.dumps(exp_config, indent=2).encode("utf-8"),
                file_name="snnqc_experiment_config.json", mime="application/json",
            )

            push_experiment_result(
                dataset_name  = st.session_state.get("dataset_name", "—"),
                model_name    = classifier_name.split("[")[0].strip(),
                n_samples     = int(X_final.shape[0]),
                n_features    = len(selected_feature_names),
                acc           = final_acc,
                f1            = final_f1,
                seed          = seed_val,
            )
            st.success("Result saved to experiment history — download from the sidebar.")
