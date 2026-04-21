import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import random
import base64
import json

from snnqc.data_loader import (
    combined_csv_template,
    dataframe_to_csv_bytes,
    dataset_summary,
    labels_csv_template,
    load_builtin_eeg_dataset as load_builtin_eeg_dataset_uncached,
    parse_combined_csv,
    parse_sample_csvs,
    read_uploaded_csv,
    sample_csv_template,
)
from snnqc.plots import feature_layout_figure, raw_and_spike_figure

# --- SKLEARN IMPORTS ---
from sklearn.metrics import accuracy_score as accuracy, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --- NEUCUBE IMPORTS ---
try:
    from neucube import Reservoir
    from neucube.encoder import Delta
    from neucube.sampler import SpikeCount
    from neucube.training import STDP
    from neucube.qfeatures import extract_features
    from neucube.qkernel import kernel_matrix, kernel
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}")
    st.info("Ensure the 'neucube' folder, 'qfeatures.py', and 'qkernel.py' are in the same directory.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SNN-QC", layout="wide", page_icon="🧠")

# --- SEED CONFIGURATION ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DATASET_STATE_KEYS = [
    "X_raw", "X", "y", "feature_names", "dataset_name", "data_ready",
    "map_initialised", "features_ready", "snn_features"
]

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .snnqc-hero {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(180deg, rgba(248, 250, 252, 0.95), rgba(241, 245, 249, 0.78));
        }
        .snnqc-hero h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin: 0 0 0.35rem 0;
            letter-spacing: 0;
        }
        .snnqc-hero p {
            color: #475569;
            font-size: 1rem;
            margin: 0;
        }
        .snnqc-stage-note {
            color: #64748b;
            font-size: 0.92rem;
        }
    </style>
    <div class="snnqc-hero">
        <h1>SNN-QC Workbench</h1>
        <p>Upload labelled time-series data, extract NeuCube-inspired spiking features, and compare classical and quantum classifiers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_help_page():
    st.markdown("## Help & App Info")
    st.write(
        "SNN-QC is a workbench for exploring time-series classification with spike encoding, "
        "NeuCube-style reservoir dynamics, spiking feature extraction, and classical or quantum classifiers."
    )

    tab_overview, tab_data, tab_pipeline, tab_results, tab_about = st.tabs(
        ["Overview", "Data Formats", "Pipeline", "Results", "About"]
    )

    with tab_overview:
        st.markdown(
            """
            ### What this app is for
            Use this app when you have labelled time-series samples and want to test whether spiking
            neural network features help separate classes.

            The built-in EEG dataset is a reference demo. Uploaded datasets can come from EEG,
            sensors, wearables, industrial monitoring, finance, or any other repeated time-series source.

            ### Current scope
            - Works best with clean, regularly sampled time-series data.
            - Every sample must have the same number of timepoints.
            - Quantum Kernel SVM currently supports exactly two selected features and two selected classes.
            - Classical SVM and Logistic Regression can use multiple features and multiple classes.
            """
        )

    with tab_data:
        st.markdown(
            """
            ### Single combined CSV
            Use one row per sample-timepoint. Include:
            - a sample ID column
            - a label column
            - an optional time/order column
            - one or more numeric feature columns

            ### Multiple sample CSVs
            Use one CSV per sample. Every sample CSV must have the same rows and columns.
            Upload a separate labels CSV with one label per sample file. Labels are matched to files
            in sorted filename order.

            ### Before running
            Check the parsed preview for samples, timepoints, feature count, and class balance.
            Uneven class balance can make cross-validation less reliable.
            """
        )

    with tab_pipeline:
        st.markdown(
            """
            ### Processing stages
            1. Data is loaded into a `samples x timepoints x features` tensor.
            2. Delta encoding converts raw values into spike trains.
            3. NeuCube reservoir simulation learns spatio-temporal activity patterns.
            4. Spike counts are pooled into SNN features.
            5. A classifier is evaluated with stratified cross-validation.

            ### Main controls
            - Delta Threshold controls spike sensitivity.
            - Reservoir C and L control reservoir connectivity.
            - STDP parameters control weight updates during spatio-temporal learning.
            - Membrane Threshold controls reservoir neuron firing.
            """
        )

    with tab_results:
        st.markdown(
            """
            ### How to read outputs
            - Accuracy shows the overall proportion of correct predictions.
            - Weighted F1 is more informative when class counts are uneven.
            - Per-class metrics show which classes are easy or hard to identify.
            - Confusion matrix shows which classes are being confused.
            - Exported SNN features can be reused in notebooks or other modelling tools.
            """
        )

    with tab_about:
        st.markdown(
            """
            ### Credits
            Original SNN-QC toolbox and research prototype developed by **Dr. Ravi Kumar Jha**
            at the Intelligent Systems Research Centre, Ulster University.

            Contact from original app: **Jha-R@ulster.ac.uk**

            This repository extends the original work into a more general-purpose Streamlit
            workbench for reusable time-series experiments.

            ### Data and materials
            The EEG dataset and NeuCube software environment are made available from Auckland
            University of Technology at https://kedri.aut.ac.nz/neucube.

            ### References
            1. Jha, R. K., Kasabov, N., Bhattacharyya, S., Coyle, D., & Prasad, G. (2025).
               A hybrid spiking neural network-quantum framework for spatio-temporal data classification:
               a case study on EEG data. EPJ Quantum Technology, 12(1), 1-23.
               https://doi.org/10.1140/epjqt/s40507-025-00443-1

            2. Kasabov, N. (2014). NeuCube: A spiking neural network architecture for mapping,
               learning and understanding of spatio-temporal brain data. Neural Networks, 52, 62-76.
               https://doi.org/10.1016/j.neunet.2014.01.006

            ### Product direction
            This workbench is being shaped into a reusable research product: clear data onboarding,
            reproducible configuration export, interpretable metrics, and downloadable feature tables.
            """
        )


def render_csv_templates():
    with st.expander("CSV Templates"):
        template_col_1, template_col_2, template_col_3 = st.columns(3)
        with template_col_1:
            st.download_button(
                "Combined CSV template",
                dataframe_to_csv_bytes(combined_csv_template()),
                file_name="snnqc_combined_template.csv",
                mime="text/csv",
            )
        with template_col_2:
            st.download_button(
                "Sample CSV template",
                dataframe_to_csv_bytes(sample_csv_template()),
                file_name="snnqc_sample_template.csv",
                mime="text/csv",
            )
        with template_col_3:
            st.download_button(
                "Labels CSV template",
                dataframe_to_csv_bytes(labels_csv_template()),
                file_name="snnqc_labels_template.csv",
                mime="text/csv",
            )


def render_dataset_preview(X_raw, y_data, feature_names):
    preview = dataset_summary(X_raw, y_data, feature_names)
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Samples", preview["samples"])
    metric_2.metric("Timepoints", preview["timepoints"])
    metric_3.metric("Features", preview["features"])
    st.write("Class balance")
    st.dataframe(
        preview["class_counts"].rename("count").reset_index().rename(columns={"index": "class"}),
        use_container_width=True,
    )


def render_loaded_dataset_summary(X, y, feature_names):
    summary = dataset_summary(X, y, feature_names)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", summary["samples"])
    col2.metric("Timepoints", summary["timepoints"])
    col3.metric("Features", summary["features"])
    col4.metric("Classes", len(summary["class_counts"]))
    st.dataframe(
        summary["class_counts"].rename("count").reset_index().rename(columns={"index": "class"}),
        use_container_width=True,
    )
    with st.expander("Feature Names"):
        st.write(", ".join(map(str, feature_names)))


def render_workflow_dashboard():
    data_ready = st.session_state.get("data_ready", False)
    map_ready = st.session_state.get("map_initialised", False)
    features_ready = st.session_state.get("features_ready", False)

    if not data_ready:
        next_step = "Load a built-in or uploaded dataset."
    elif not map_ready:
        next_step = "Inspect encoded signals, then initialise the NeuCube mapping."
    elif not features_ready:
        next_step = "Run the NeuCube simulation to extract SNN features."
    else:
        next_step = "Run classification and export the metrics/features."

    st.info(f"Next step: {next_step}")

    tab_data, tab_encoding, tab_simulation, tab_features, tab_results = st.tabs(
        ["Data", "Encoding", "Simulation", "Features", "Results"]
    )

    with tab_data:
        status = "Ready" if data_ready else "Not loaded"
        st.metric("Dataset", status)
        if data_ready:
            X = st.session_state["X"]
            y = st.session_state["y"]
            feature_names = st.session_state["feature_names"]
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Samples", X.shape[0])
            col_b.metric("Timepoints", X.shape[1])
            col_c.metric("Features", X.shape[2])
            col_d.metric("Classes", pd.Series(y).nunique())
            st.caption(f"Loaded dataset: {st.session_state.get('dataset_name', 'Loaded dataset')}")
            st.caption(f"First features: {', '.join(map(str, feature_names[:8]))}")
        else:
            st.caption("Start below by choosing a dataset source and loading data.")

    with tab_encoding:
        status = "Encoded" if data_ready else "Waiting for data"
        st.metric("Spike Encoding", status)
        st.caption(f"Current Delta threshold: {threshold_val}")
        if data_ready:
            X = st.session_state["X"]
            spike_density = float(np.count_nonzero(X.numpy()) / X.numel())
            st.metric("Spike Density", f"{spike_density:.2%}")
        else:
            st.caption("Encoding preview appears after data loading.")

    with tab_simulation:
        status = "Features extracted" if features_ready else "Mapping ready" if map_ready else "Waiting"
        st.metric("NeuCube Simulation", status)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Reservoir C", res_c)
        col_b.metric("Reservoir L", res_l)
        col_c.metric("Membrane Threshold", mem_thr_val)

    with tab_features:
        status = "Ready" if features_ready else "Not available"
        st.metric("SNN Features", status)
        if features_ready:
            st.write(f"Extracted feature matrix: `{st.session_state['snn_features'].shape}`")
            st.caption("Use the Features section below to download the extracted feature table.")
        else:
            st.caption("Run the NeuCube simulation to generate features.")

    with tab_results:
        status = "Ready to classify" if features_ready else "Waiting for features"
        st.metric("Classification", status)
        st.caption("Classical classifiers support multiclass use. Quantum Kernel SVM is currently binary and two-feature.")
        if data_ready:
            st.download_button(
                "Download Experiment Config",
                json.dumps(current_experiment_config(), indent=2).encode("utf-8"),
                file_name="snnqc_experiment_config.json",
                mime="application/json",
            )


def current_experiment_config():
    config = {
        "app": "SNN-QC Workbench",
        "dataset": st.session_state.get("dataset_name"),
        "data_ready": st.session_state.get("data_ready", False),
        "map_initialised": st.session_state.get("map_initialised", False),
        "features_ready": st.session_state.get("features_ready", False),
        "hyperparameters": {
            "delta_threshold": threshold_val,
            "reservoir_c": res_c,
            "reservoir_l": res_l,
            "stdp_a_pos": stdp_pos,
            "stdp_a_neg": stdp_neg,
            "membrane_threshold": mem_thr_val,
            "svm_c": svm_c,
            "cv_folds": k_folds,
        },
    }

    if st.session_state.get("data_ready", False):
        X = st.session_state["X"]
        y = st.session_state["y"]
        config["data"] = {
            "shape": list(X.shape),
            "classes": [str(value) for value in sorted(pd.Series(y).dropna().unique(), key=str)],
            "feature_names": [str(value) for value in st.session_state.get("feature_names", [])],
        }

    if st.session_state.get("features_ready", False):
        config["snn_features"] = {
            "shape": list(st.session_state["snn_features"].shape),
        }

    return config


page = st.sidebar.radio("Page", ["Workbench", "Help & App Info"])
if page == "Help & App Info":
    render_help_page()
    st.stop()


# ==========================================
# 1. SIDEBAR PARAMETERS
# ==========================================
st.sidebar.header("1. NeuCube Hyperparameters")
threshold_val = st.sidebar.slider("Delta Threshold", 0.1, 1.5, 0.8, 0.05)
res_c = st.sidebar.slider("Reservoir C", 0.1, 1.0, 0.4, 0.05)
res_l = st.sidebar.slider("Reservoir L", 0.01, 1.0, 0.169, 0.001)
stdp_pos = st.sidebar.number_input("STDP a_pos", value=0.001, format="%.4f")
stdp_neg = st.sidebar.number_input("STDP a_neg", value=-0.01, format="%.4f")
mem_thr_val = st.sidebar.number_input("Membrane Threshold", value=0.01, format="%.3f")

st.sidebar.header("2. Model Settings")
svm_c = st.sidebar.number_input("Regularization (C)", value=1.0)
k_folds = st.sidebar.slider("CV Folds", 2, 10, 5)

# --- SIDEBAR FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info(
    "**SNN-QC Workbench**\n\n"
    "Reusable spiking-feature classification for labelled time-series datasets.\n\n"
    "Original toolbox: **Dr. Ravi Kumar Jha**, Ulster University."
)

render_workflow_dashboard()

# ==========================================
# 2. DATA LOADING (Cached)
# ==========================================
@st.cache_data
def load_builtin_eeg_dataset():
    return load_builtin_eeg_dataset_uncached()

def encode_dataset(X_raw, thresh):
    encoder = Delta(threshold=thresh)
    return encoder.encode_dataset(X_raw)

def reset_dataset_state():
    for key in DATASET_STATE_KEYS:
        st.session_state.pop(key, None)

# ==========================================
# 3. MAIN EXECUTION FLOW
# ==========================================

st.markdown("""
    <h2 style='font-size: 24px;'> Data Setup </h2>
""", unsafe_allow_html=True)

dataset_source = st.radio(
    "Dataset Source",
    ["Built-in EEG demo", "Single combined CSV", "Multiple sample CSVs"],
    horizontal=True,
)

render_csv_templates()

combined_df = None
combined_config = {}
sample_files = []
sample_labels_df = None
sample_has_header = True

if dataset_source == "Built-in EEG demo":
    st.caption("Uses the bundled wrist movement EEG dataset: 60 trials, 128 timepoints, 14 channels.")

elif dataset_source == "Single combined CSV":
    st.caption("Expected format: one row per sample-timepoint, with columns for sample ID, label, and numeric features.")
    combined_file = st.file_uploader("Upload combined CSV", type=["csv"], key="combined_csv")
    combined_has_header = st.checkbox("Combined CSV has a header row", value=True)

    if combined_file is not None:
        combined_df = read_uploaded_csv(combined_file, combined_has_header)
        st.dataframe(combined_df.head(20), use_container_width=True)

        columns = [str(col) for col in combined_df.columns]
        combined_df.columns = columns
        default_label_index = len(columns) - 1

        col_cfg_1, col_cfg_2, col_cfg_3 = st.columns(3)
        with col_cfg_1:
            sample_col = st.selectbox("Sample ID column", columns)
        with col_cfg_2:
            label_col = st.selectbox("Label column", columns, index=default_label_index)
        with col_cfg_3:
            time_options = ["(none)"] + columns
            time_col = st.selectbox("Time/order column", time_options)

        excluded = {sample_col, label_col}
        if time_col != "(none)":
            excluded.add(time_col)
        default_features = [col for col in columns if col not in excluded]
        feature_cols = st.multiselect("Feature columns", columns, default=default_features)
        combined_config = {
            "sample_col": sample_col,
            "label_col": label_col,
            "time_col": time_col,
            "feature_cols": feature_cols,
        }

        preview_X, preview_y, preview_features, preview_err = parse_combined_csv(
            combined_df, sample_col, label_col, time_col, feature_cols
        )
        if preview_err:
            st.warning(preview_err)
        else:
            render_dataset_preview(preview_X, preview_y, preview_features)

elif dataset_source == "Multiple sample CSVs":
    st.caption("Expected format: each sample CSV is timepoints x features. Labels CSV must contain one label per sample file, in sorted filename order.")
    sample_has_header = st.checkbox("Sample CSVs have a header row", value=True)
    sample_files = st.file_uploader("Upload sample CSV files", type=["csv"], accept_multiple_files=True, key="sample_csvs")
    sample_labels_file = st.file_uploader("Upload labels CSV", type=["csv"], key="sample_labels_csv")
    labels_has_header = st.checkbox("Labels CSV has a header row", value=False)

    if sample_files:
        sorted_names = [file_obj.name for file_obj in sorted(sample_files, key=lambda file_obj: file_obj.name)]
        st.write("Sample order used for labels:", ", ".join(sorted_names[:10]) + (" ..." if len(sorted_names) > 10 else ""))
    if sample_labels_file is not None:
        sample_labels_df = read_uploaded_csv(sample_labels_file, labels_has_header)
        st.dataframe(sample_labels_df.head(20), use_container_width=True)

    if sample_files and sample_labels_df is not None:
        preview_X, preview_y, preview_features, preview_err = parse_sample_csvs(
            sample_files, sample_labels_df, sample_has_header
        )
        if preview_err:
            st.warning(preview_err)
        else:
            render_dataset_preview(preview_X, preview_y, preview_features)

if st.button("Load & Encode Data"):
    with st.spinner(f"Loading files and applying Delta Encoding (Thresh={threshold_val})..."):
        if dataset_source == "Built-in EEG demo":
            X_raw, y_data, feature_names, err = load_builtin_eeg_dataset()
            dataset_name = "Built-in EEG demo"
        elif dataset_source == "Single combined CSV":
            X_raw, y_data, feature_names, err = parse_combined_csv(
                combined_df,
                combined_config.get("sample_col"),
                combined_config.get("label_col"),
                combined_config.get("time_col"),
                combined_config.get("feature_cols", []),
            )
            dataset_name = "Uploaded combined CSV"
        else:
            X_raw, y_data, feature_names, err = parse_sample_csvs(sample_files, sample_labels_df, sample_has_header)
            dataset_name = "Uploaded sample CSVs"
        
        if err:
            st.error(err)
        else:
            reset_dataset_state()
            X_encoded = encode_dataset(X_raw, threshold_val)
            st.success("Data loaded and encoded successfully.")
            render_loaded_dataset_summary(X_encoded, y_data, feature_names)
            
            st.session_state['X_raw'] = X_raw
            st.session_state['X'] = X_encoded 
            st.session_state['y'] = y_data
            st.session_state['feature_names'] = feature_names
            st.session_state['dataset_name'] = dataset_name
            st.session_state['data_ready'] = True

# Check if data is ready
if st.session_state.get('data_ready', False):
    X_raw = st.session_state['X_raw']
    X = st.session_state['X']
    y = st.session_state['y']
    feature_names = st.session_state.get('feature_names', [f"Feature {idx + 1}" for idx in range(X.shape[2])])

    st.sidebar.header("3. Features Selection")
    feat_name_1 = st.sidebar.selectbox("Feature 1 (Channel)", feature_names, index=0)
    feat_2_default = min(4, len(feature_names) - 1)
    feat_name_2 = st.sidebar.selectbox("Feature 2 (Channel)", feature_names, index=feat_2_default)
    feat_idx_1 = feature_names.index(feat_name_1)
    feat_idx_2 = feature_names.index(feat_name_2)

    with st.expander("Dataset Summary", expanded=True):
        st.write(f"**Dataset:** {st.session_state.get('dataset_name', 'Loaded dataset')}")
        render_loaded_dataset_summary(X, y, feature_names)
    
    with st.expander("Raw Signals & Spikes", expanded=True):
        st.caption("Encode Spikes")
        
        col_viz_1, col_viz_2 = st.columns(2)
        
        with col_viz_1:
            sel_sample = st.slider("Select Trial", 0, len(X)-1, 0)
            
        with col_viz_2:
            sel_channel = st.selectbox("Select Feature (Channel)", feature_names, index=0, key="viz_chan_sel")
        
        ch_idx_viz = feature_names.index(sel_channel)
        
        raw_sig = X_raw[sel_sample, :, ch_idx_viz].numpy()
        spike_sig = X[sel_sample, :, ch_idx_viz].numpy()

        st.pyplot(raw_and_spike_figure(raw_sig, spike_sig, sel_channel, sel_sample))

    st.divider()
    
    st.markdown("""
        <h2 style='font-size: 24px;'> NeuCube Spatio-temporal Learning </h2>
    """, unsafe_allow_html=True)
    
    if st.button("NeuCube Initialisation & Mapping"):
        st.session_state['map_initialised'] = True

    if st.session_state.get('map_initialised', False):
        with st.expander("Schematic Brain Template", expanded=True):
            col_ctrl_1, col_ctrl_2 = st.columns([1, 4])
            with col_ctrl_1:
                clean_view = st.toggle("Grid & Axes", value=True)

            st.plotly_chart(feature_layout_figure(feature_names, clean_view), use_container_width=True)


    if st.session_state.get('map_initialised', False):
        st.divider()
        
        col_sim_1, col_sim_2 = st.columns([1, 3])
        
        with col_sim_1:
            st.write("") 
            st.write("")
            run_sim = st.button("▶ Run NeuCube Simulation")

        if run_sim:
            with col_sim_2:
                brain_placeholder = st.empty()
                
                try:
                    file_path = "brain_activity.gif"
                    with open(file_path, "rb") as f:
                        contents = f.read()
                        data_url = base64.b64encode(contents).decode("utf-8")

                    brain_placeholder.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" width="300" style="border-radius: 10px;">',
                        unsafe_allow_html=True,
                    )
                except FileNotFoundError:
                    brain_placeholder.info("🧠 NeuCube is thinking... (GIF not found)")

                with st.spinner("Spatio-temporal Training..."):
                    res = Reservoir(inputs=X.shape[2], c=res_c, l=res_l)
                    learning_rule = STDP(a_pos=stdp_pos, a_neg=stdp_neg)
                    sam = SpikeCount()

                    s_act_all = res.simulate(
                        X,
                        train=True,                  
                        learning_rule=learning_rule,
                        mem_thr=mem_thr_val,                
                        refractory_period=5,
                        verbose=True 
                    )

                    state_vectors_all = sam.sample(s_act_all)
                    snn_features = extract_features(state_vectors_all, res.w_in)
                    
                    st.session_state['snn_features'] = snn_features
                    st.session_state['features_ready'] = True
                
                brain_placeholder.empty()
            
            st.success("Simulation Complete!")


    if st.session_state.get('features_ready', False):
        st.divider()
        st.markdown("""
                <h2 style='font-size: 24px;'> Spiking Feature Extraction </h2>
        """, unsafe_allow_html=True)
        st.caption("Trained spike frequency state vectors.")
        
        st.write(f"**Extracted Features Shape:** {st.session_state['snn_features'].shape}")

        features_df = pd.DataFrame(st.session_state['snn_features'], columns=feature_names)
        features_df.insert(0, "label", y)
        features_df.insert(0, "sample_index", np.arange(len(features_df)))
        st.download_button(
            "Download SNN Features",
            dataframe_to_csv_bytes(features_df),
            file_name="snn_features.csv",
            mime="text/csv",
        )
        
    if st.session_state.get('features_ready', False):
        snn_features = st.session_state['snn_features']
        
        st.divider()
        st.markdown("""
                <h2 style='font-size: 24px;'> Classification </h2>
        """, unsafe_allow_html=True)
        
        available_classes = sorted(pd.Series(y).dropna().unique(), key=str)
        classifier_name = st.selectbox(
            "Classifier",
            ["Quantum Kernel SVM", "Classical SVM", "Logistic Regression"],
        )

        if len(available_classes) < 2:
            st.warning("At least two classes are required for classification.")
            st.stop()

        default_class_selection = available_classes[:2] if classifier_name == "Quantum Kernel SVM" else available_classes
        selected_classes = st.multiselect(
            "Classes for classification",
            available_classes,
            default=default_class_selection,
        )

        if classifier_name == "Quantum Kernel SVM" and len(selected_classes) != 2:
            st.warning("Choose exactly two classes to run the current 2-feature quantum kernel classifier.")
            st.stop()
        if classifier_name != "Quantum Kernel SVM" and len(selected_classes) < 2:
            st.warning("Choose at least two classes.")
            st.stop()

        class_mask = np.isin(y, selected_classes)
        filtered_features = snn_features[class_mask]
        y_final = y[class_mask]

        if classifier_name == "Quantum Kernel SVM":
            feature_indices = [int(feat_idx_1), int(feat_idx_2)]
            selected_feature_names = [feat_name_1, feat_name_2]
        else:
            default_feature_names = feature_names if len(feature_names) <= 20 else feature_names[:20]
            selected_feature_names = st.multiselect(
                "Features for classification",
                feature_names,
                default=default_feature_names,
            )
            if not selected_feature_names:
                st.warning("Choose at least one feature.")
                st.stop()
            feature_indices = [feature_names.index(name) for name in selected_feature_names]

        X_final = filtered_features[:, feature_indices]
        
        col1, col2 = st.columns(2)
        col1.write(f"**Classifier Input Shape:** {X_final.shape}")
        
        col2.write(f"**Selected Features:** {', '.join(map(str, selected_feature_names))}")

        if classifier_name == "Quantum Kernel SVM":
            with st.expander("View Quantum Kernel Circuit"):
                try:
                    fig, ax = qml.draw_mpl(kernel)(X_final[0], X_final[1])
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Circuit visualization error: {e}")

        if st.button("Run Cross-Validation"):
            class_counts = pd.Series(y_final).value_counts()
            effective_folds = min(k_folds, int(class_counts.min()))
            if effective_folds < 2:
                st.warning("Each selected class needs at least two samples for cross-validation.")
                st.stop()

            kf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=int(SEED))
            y_total2, pred_total2 = [], []

            if classifier_name == "Quantum Kernel SVM":
                classifier = SVC(kernel=kernel_matrix, C=svm_c)
            elif classifier_name == "Classical SVM":
                classifier = SVC(kernel="rbf", C=svm_c)
            else:
                classifier = LogisticRegression(max_iter=1000)

            progress_bar = st.progress(0)
            status_txt = st.empty()

            for i, (train_index, test_index) in enumerate(kf.split(X_final, y_final)):
                status_txt.text(f"Running Fold {i+1}/{effective_folds}...")
                
                X_train, X_test = X_final[train_index], X_final[test_index]
                y_train, y_test = y_final[train_index], y_final[test_index]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                classifier.fit(X_train, y_train)
                pred2 = classifier.predict(X_test)

                y_total2.extend(y_test)
                pred_total2.extend(pred2)
                
                progress_bar.progress((i + 1) / effective_folds)

            status_txt.text("Validation Complete.")
            
            final_acc = accuracy(y_total2, pred_total2)
            final_f1 = f1_score(y_total2, pred_total2, average="weighted", zero_division=0)
            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric("Accuracy", f"{final_acc:.2%}")
            metric_col_2.metric("Weighted F1", f"{final_f1:.2%}")

            report = classification_report(
                y_total2,
                pred_total2,
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).T
            st.markdown("##### Per-Class Metrics")
            st.dataframe(report_df, use_container_width=True)
            st.download_button(
                "Download Classification Metrics",
                dataframe_to_csv_bytes(report_df.reset_index().rename(columns={"index": "class"})),
                file_name="classification_metrics.csv",
                mime="text/csv",
            )
            
            st.markdown("##### Confusion Matrix")
            
            cm = confusion_matrix(y_total2, pred_total2)
            unique_labels = sorted(list(set(y_total2)), key=str)
            
            col_cm_1, col_cm_2 = st.columns([1, 2])
            
            with col_cm_1:
                fig_cm, ax_cm = plt.subplots(figsize=(4, 2))
                
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
                disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
                
                ax_cm.set_title("Confusion Matrix", fontsize=6)
                ax_cm.set_xlabel("Predicted label", fontsize=6)
                ax_cm.set_ylabel("True label", fontsize=6)
                ax_cm.tick_params(axis='both', labelsize=6)
                
                st.pyplot(fig_cm, width='content')
