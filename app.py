import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pennylane as qml
import random
import plotly.graph_objects as go
import base64

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

st.title("🧠 SNN-QC: Spiking Neural Network-Quantum Computational Toolbox")
st.markdown("""
    <div style='text-align: center; color: #1E90FF; font-size: 18px; font-weight: bold;'>
        [Pipeline: Data Upload & Spike Encoding &rarr; NeuCube Spatio-temporal Learning &rarr; Spiking Feature Extraction &rarr; Quantum Kernel Classification]
    </div>
    """, unsafe_allow_html=True)


def render_help_page():
    st.markdown("## Help & App Info")
    st.write(
        "SNN-QC is a workbench for exploring time-series classification with spike encoding, "
        "NeuCube-style reservoir dynamics, spiking feature extraction, and classical or quantum classifiers."
    )

    tab_overview, tab_data, tab_pipeline, tab_results = st.tabs(
        ["Overview", "Data Formats", "Pipeline", "Results"]
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
    "**© 2025 Intelligent Systems Research Centre**\n\n"
    "*Ulster University*\n\n"
    "Developed by: **Ravi Kumar Jha**\n\n"
    "Contact: Jha-R@ulster.ac.uk"
)

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

def load_sensor_locations():
    """
    Uses eeg_mapping to find the specific XYZ coordinates.
    STRICTLY enforces 14 points to match the default EEG feature list.
    """
    base_path = './example_data/wrist_movement_eeg/'
    coord_path = os.path.join(base_path, 'brain_coordinates.csv')
    map_path = os.path.join(base_path, 'eeg_mapping.csv')
    
    if os.path.exists(coord_path) and os.path.exists(map_path):
        # 1. Load all reservoir coordinates (e.g., 1471 points)
        all_coords = pd.read_csv(coord_path, header=None).values
        
        # 2. Load the mapping indices
        mapping_indices = pd.read_csv(map_path, header=None).values.flatten().astype(int)
        
        # --- FIX 1: FORCE EXACTLY 14 INDICES ---
        # If the file has 15, we take the first 14. 
        # If the file has 14, this does nothing (safe).
        if len(mapping_indices) > 14:
            mapping_indices = mapping_indices[:14]
            
        # 3. Select the rows
        channel_coords = all_coords[mapping_indices]
        
        return channel_coords
    else:
        return None

# ==========================================
# 3. MAIN EXECUTION FLOW
# ==========================================

st.markdown("""
    <h2 style='font-size: 24px;'> Data Upload & Spike Encoding </h2>
""", unsafe_allow_html=True)

dataset_source = st.radio(
    "Dataset Source",
    ["Built-in EEG demo", "Single combined CSV", "Multiple sample CSVs"],
    horizontal=True,
)

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
            preview = dataset_summary(preview_X, preview_y, preview_features)
            metric_1, metric_2, metric_3 = st.columns(3)
            metric_1.metric("Samples", preview["samples"])
            metric_2.metric("Timepoints", preview["timepoints"])
            metric_3.metric("Features", preview["features"])
            st.write("Class balance")
            st.dataframe(
                preview["class_counts"].rename("count").reset_index().rename(columns={"index": "class"}),
                use_container_width=True,
            )

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
            preview = dataset_summary(preview_X, preview_y, preview_features)
            metric_1, metric_2, metric_3 = st.columns(3)
            metric_1.metric("Samples", preview["samples"])
            metric_2.metric("Timepoints", preview["timepoints"])
            metric_3.metric("Features", preview["features"])
            st.write("Class balance")
            st.dataframe(
                preview["class_counts"].rename("count").reset_index().rename(columns={"index": "class"}),
                use_container_width=True,
            )

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
            st.success(f"Data Loaded Successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 10px; border-radius: 5px;">
                    <p style="margin:0; font-size: 20px; color: #9da3a8;">Input Shape</p>
                    <p style="margin:0; font-size: 20px; font-weight: bold; color: white;">{X_encoded.shape}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 10px; border-radius: 5px;">
                    <p style="margin:0; font-size: 20px; color: #9da3a8;">Total Labels</p>
                    <p style="margin:0; font-size: 20px; font-weight: bold; color: white;">{len(y_data)}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 10px; border-radius: 5px;">
                    <p style="margin:0; font-size: 20px; color: #9da3a8;">Features</p>
                    <p style="margin:0; font-size: 20px; font-weight: bold; color: white;">{len(feature_names)}</p>
                </div>
                """, unsafe_allow_html=True)
            
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
        st.write(f"**Samples x Timepoints x Features:** {tuple(X.shape)}")
        st.write(f"**Classes:** {', '.join(map(str, sorted(pd.Series(y).dropna().unique())))}")
        st.write(f"**Feature names:** {', '.join(map(str, feature_names))}")
    
        # --- UPDATED: VISUALIZATION SECTION ---
    with st.expander("Raw Signals & Spikes", expanded=True):
        st.caption("Encode Spikes")
        
        # --- SELECTORS ---
        col_viz_1, col_viz_2 = st.columns(2)
        
        with col_viz_1:
            # 1. Select Sample
            sel_sample = st.slider("Select Trial", 0, len(X)-1, 0)
            
        with col_viz_2:
            # 2. Select Feature (Channel) - THIS IS THE NEW DROP DOWN
            sel_channel = st.selectbox("Select Feature (Channel)", feature_names, index=0, key="viz_chan_sel")
        
        # Convert the selected Name (e.g., "T7") to Index (e.g., 4)
        ch_idx_viz = feature_names.index(sel_channel)
        
        # --- PREPARE DATA FOR PLOTTING ---
        # Raw Data (Continuous values)
        raw_sig = X_raw[sel_sample, :, ch_idx_viz].numpy()
        
        # Encoded Data (Discrete Spikes: -1, 0, 1)
        spike_sig = X[sel_sample, :, ch_idx_viz].numpy()
        
        # --- PLOTTING LOGIC ---
        # Create two subplots sharing the X-axis (Time)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 3.5), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Raw Signal
        ax1.plot(raw_sig, color='#1E90FF', linewidth=1.0, label='Raw Input')
        ax1.set_ylabel(" Signal Values")
        ax1.set_title(f"Feature {sel_channel} (Trial {sel_sample})")
        ax1.grid(True, alpha=0.5)
        ax1.legend(loc='upper right')
        
        # Plot 2: Spikes 
        # Using a stem plot to clearly show discrete events
        markerline, stemlines, baseline = ax2.stem(
            np.arange(len(spike_sig)), 
            spike_sig, 
            linefmt='#1E90FF', 
            markerfmt=' ', # You can change 'o' to ' ' if you want bars only (no dots)
            basefmt='gray'
        )
        
        # Style the lines to be thin and clean
        plt.setp(stemlines, 'linewidth', 1.2)
        #plt.setp(markerline, 'markersize', 2)
        plt.setp(baseline, 'linewidth', 0.1, 'alpha', 0.1)
        
        ax2.set_ylabel("Spike State")
        ax2.set_xlabel("Time Steps")
        
        # --- KEY CHANGES HERE ---
        ax2.set_yticks([])   # Only show 0 and 1 on the axis
        #ax2.set_ylim(-0.1, 1.25) # Crop out the negative space (-1) completely
        # ------------------------
        
        ax2.set_title("Output Spikes") 
        #ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()
    
    #st.header("NeuCube Spatio-temporal Learning")
    st.markdown("""
        <h2 style='font-size: 24px;'> NeuCube Spatio-temporal Learning </h2>
    """, unsafe_allow_html=True)
    
    #SENSOR VISUALIZATION
    # Define Standard 10-20 Coordinates
    if st.button("NeuCube Initialisation & Mapping"):
        st.session_state['map_initialised'] = True
    # Render the Map only if initialized
    if st.session_state.get('map_initialised', False):
        
        # --- DATA PREPARATION ---
        STANDARD_10_20 = {
            "AF3": [-30, 50, 30], "O2": [20, -90, 10],
            "F7": [-50, 30, 0], "P8": [60, -60, 0],
            "F3": [-40, 30, 40], "T8": [70, -20, 0],
            "FC5": [-60, 0, 30], "FC6": [60, 0, 30],
            "T7": [-70, -20, 0], "F4":  [40, 30, 40],
            "P7": [-60, -60, 0], "F8": [50, 30, 0],
            "O1": [-20, -90, 10], "AF4": [30, 50, 30]
        }

        plot_coords = []
        plot_names = []
        eeg_matches = 0
        
        for raw_name in feature_names:
            clean_name = raw_name.replace('*', '') 
            if clean_name in STANDARD_10_20:
                plot_coords.append(STANDARD_10_20[clean_name])
                plot_names.append(raw_name)
                eeg_matches += 1
            else:
                plot_coords.append([0,0,0])
                plot_names.append(raw_name)
                
        plot_coords = np.array(plot_coords)
        use_eeg_template = eeg_matches > 0

        if not use_eeg_template:
            angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
            plot_coords = np.column_stack([
                70 * np.cos(angles),
                70 * np.sin(angles),
                np.zeros(len(feature_names)),
            ])

        # --- VIEW CONTROL ---
        with st.expander("Schematic Brain Template", expanded=True):
            col_ctrl_1, col_ctrl_2 = st.columns([1, 4])
            with col_ctrl_1:
                clean_view = st.toggle("Grid & Axes", value=True)

            # --- PLOTTING ---
            fig_3d = go.Figure()

            # Electrodes
            fig_3d.add_trace(go.Scatter3d(
                x=plot_coords[:, 0], y=plot_coords[:, 1], z=plot_coords[:, 2],
                mode='markers+text',
                text=plot_names,
                textposition="top center",
                textfont=dict(family="Arial Black", size=12, color="white"), 
                marker=dict(size=12, color='#FF4B4B', opacity=1.0, line=dict(width=2, color='white')),
                name="Electrodes"
            ))

            # Ghost Head Model
            if use_eeg_template:
                phi = np.linspace(0, 2*np.pi, 20)
                theta = np.linspace(0, np.pi, 10)
                phi, theta = np.meshgrid(phi, theta)
                r = 90 
                x_sphere = r * np.sin(theta) * np.cos(phi)
                y_sphere = r * np.sin(theta) * np.sin(phi)
                z_sphere = r * np.cos(theta) - 10 
            
                fig_3d.add_trace(go.Mesh3d(
                    x=x_sphere.flatten(), y=y_sphere.flatten(), z=z_sphere.flatten(),
                    color='gray', opacity=0.2, name='Head Model', alphahull=0 
                ))

            # Layout Logic
            if clean_view:
                grid_status = False
                axis_range = [-100, 100]
            else:
                grid_status = True
                axis_range = None

            fig_3d.update_layout(
                title="Schematic Head Model Visualisation" if use_eeg_template else "Generic Feature Layout",
                scene=dict(
                    xaxis=dict(range=axis_range, showgrid=grid_status, zeroline=grid_status, showticklabels=grid_status, title='X' if grid_status else ''),
                    yaxis=dict(range=axis_range, showgrid=grid_status, zeroline=grid_status, showticklabels=grid_status, title='Y' if grid_status else ''),
                    zaxis=dict(range=axis_range, showgrid=grid_status, zeroline=grid_status, showticklabels=grid_status, title='Z' if grid_status else ''),
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700,
                showlegend=False
            )

            st.plotly_chart(fig_3d, use_container_width=True)


    # ==========================================
    # B. SIMULATION LOGIC (COMES SECOND)
    # ==========================================
    
    # We only show the Run Simulation button if the map has been initialized
    if st.session_state.get('map_initialised', False):
        st.divider() # Visual separation
        
        col_sim_1, col_sim_2 = st.columns([1, 3])
        
        with col_sim_1:
            # Add some vertical padding so the button aligns nicely
            st.write("") 
            st.write("")
            run_sim = st.button("▶ Run NeuCube Simulation")

        if run_sim:
            with col_sim_2:
                # 1. Create a placeholder for the "Thinking Brain" animation
                brain_placeholder = st.empty()
                
                # 2. Display an animated GIF in the placeholder
                # Updated parameter: use_container_width instead of use_column_width
                try:
                    file_path = "brain_activity.gif" # Your filename
                    with open(file_path, "rb") as f:
                        contents = f.read()
                        data_url = base64.b64encode(contents).decode("utf-8")

                    # Inject HTML to force the GIF to play
                    brain_placeholder.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" width="300" style="border-radius: 10px;">',
                        unsafe_allow_html=True,
                    )
                except FileNotFoundError:
                    brain_placeholder.info("🧠 NeuCube is thinking... (GIF not found)")

                # 3. Run the heavy computation
                with st.spinner("Spatio-temporal Training..."):
                    
                    # Initialize Reservoir
                    res = Reservoir(inputs=X.shape[2], c=res_c, l=res_l)
                    learning_rule = STDP(a_pos=stdp_pos, a_neg=stdp_neg)
                    sam = SpikeCount()

                    # Simulate
                    s_act_all = res.simulate(
                        X,
                        train=True,                  
                        learning_rule=learning_rule,
                        mem_thr=mem_thr_val,                
                        refractory_period=5,
                        verbose=True 
                    )

                    # B. Extraction (Step 3 Logic - done here to save state)
                    state_vectors_all = sam.sample(s_act_all)
                    snn_features = extract_features(state_vectors_all, res.w_in)
                    
                    # Save everything
                    st.session_state['snn_features'] = snn_features
                    st.session_state['features_ready'] = True
                
                brain_placeholder.empty()
            
            st.success("Simulation Complete!")


    # ==========================================
    # STEP 3: FEATURES EXTRACTION (Visual Section)
    # ==========================================
    
    # This block is UN-INDENTED so it sits outside the button logic
    if st.session_state.get('features_ready', False):
        st.divider()
        #st.header("Spiking Feature Extraction")
        st.markdown("""
                <h2 style='font-size: 24px;'> Spiking Feature Extraction </h2>
        """, unsafe_allow_html=True)
        st.caption("Trained spike frequency state vectors.")
        
        # Display the Shape (The proof that Step 3 is done)
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
        
        # Optional: Add a plot here if you want to visualize the features
        ##with st.expander("Inspect Feature Vector"):
            ##st.bar_chart(st.session_state['snn_features'][0])
             

    # Check if features are ready
    if st.session_state.get('features_ready', False):
        snn_features = st.session_state['snn_features']
        
        st.divider()
        #st.header("Classification")
        st.markdown("""
                <h2 style='font-size: 24px;'> Classification </h2>
        """, unsafe_allow_html=True)
        
        # 1. Prepare Data
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
        
        # Show Names in UI
        col2.write(f"**Selected Features:** {', '.join(map(str, selected_feature_names))}")

        # 2. Visualise Quantum Circuit
        if classifier_name == "Quantum Kernel SVM":
            with st.expander("View Quantum Kernel Circuit"):
                try:
                    fig, ax = qml.draw_mpl(kernel)(X_final[0], X_final[1])
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Circuit visualization error: {e}")

        # 3. Classification
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
            
            # --- FINAL METRICS ---
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
            
            # --- CONFUSION MATRIX ---
            st.markdown("##### Confusion Matrix")   # smaller than subheader
            
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
                
                
st.divider()

with st.expander("📂 Data & Material"):
    st.markdown("""
    :blue[👉 **Data Availability:**]
    The EEG dataset and NeuCube software environment are kindly made available from the Auckland University of Technology at: [https://kedri.aut.ac.nz/neucube].
 
    :blue[👉 **Code Availability:**]
    The complete source code for the **SNN-QC** toolbox is kindly made available in the GitHub at: [https://github.com/ravik-jha/SNN-QC].  
    """)                
               
# --- REFERENCES SECTION ---
st.divider()

with st.expander("📚 References & Acknowledgement"):
    st.markdown("""
    :blue[**References**]
    
    [1]. Jha, R. K., Kasabov, N., Bhattacharyya, S., Coyle, D., & Prasad, G. (2025). A hybrid spiking neural network-quantum framework for spatio-temporal data classification: a case study on EEG data. 
      EPJ Quantum Technology, 12(1), 1-23. [https://doi.org/10.1140/epjqt/s40507-025-00443-1]
      
    [2]. Kasabov, N. (2014). NeuCube: A spiking neural network architecture for mapping, learning and understanding of spatio-temporal brain data. Neural networks, 52, 62-76. [https://doi.org/10.1016/j.neunet.2014.01.006] 
    
    :blue[**Acknowledgement**]
    
    This toolbox demonstration was developed as part of the research conducted at the Intelligent Systems Research Centre, Ulster University. 
    It is intended solely for academic research and demonstration purposes. This is a foundational prototype; all rights regarding future development and commercial 
    utilization are reserved and subject to copyright.

    """)
