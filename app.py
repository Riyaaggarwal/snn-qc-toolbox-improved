import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pennylane as qml
import random
from tqdm import tqdm
import plotly.graph_objects as go
import base64

# --- SKLEARN IMPORTS ---
from sklearn.metrics import accuracy_score as accuracy, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --- NEUCUBE IMPORTS ---
try:
    from neucube import Reservoir
    from neucube.encoder import Delta
    from neucube.validation import Pipeline
    from neucube.sampler import SpikeCount
    from neucube.training import STDP
    from neucube.sampler.channel_sampler import ChannelContributionSampler
    from neucube.visualise import spike_raster, plot_connections
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

DEFAULT_FEATURE_NAMES = [
    "AF3*", "F7*", "F3*", "FC5*", "T7*", "P7*", "O1*", 
    "O2*", "P8*", "T8*", "FC6*", "F4*", "F8*", "AF4*"
]
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
    """Loads the bundled EEG demo dataset as a raw tensor."""
    base_path = './example_data/wrist_movement_eeg/'
    
    if not os.path.exists(base_path):
        return None, None, DEFAULT_FEATURE_NAMES, f"Path not found: {base_path}"

    filenameslist = ['sam'+str(idx)+'_eeg.csv' for idx in range(1,61)]
    dfs = []
    
    try:
        for filename in filenameslist:
            full_path = os.path.join(base_path, filename)
            dfs.append(pd.read_csv(full_path, header=None))
    except FileNotFoundError as e:
        return None, None, DEFAULT_FEATURE_NAMES, f"Missing File: {e}"

    fulldf = pd.concat(dfs)
    labels_path = os.path.join(base_path, 'tar_class_labels.csv')
    labels = pd.read_csv(labels_path, header=None)
    y_all = labels.values.flatten()

    X_raw = torch.tensor(fulldf.values.reshape(60, 128, 14), dtype=torch.float32)
    return X_raw, y_all, DEFAULT_FEATURE_NAMES, None

def encode_dataset(X_raw, thresh):
    encoder = Delta(threshold=thresh)
    return encoder.encode_dataset(X_raw)

def reset_dataset_state():
    for key in DATASET_STATE_KEYS:
        st.session_state.pop(key, None)

def read_uploaded_csv(uploaded_file, has_header):
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    header = 0 if has_header else None
    return pd.read_csv(uploaded_file, header=header)

def parse_combined_csv(df, sample_col, label_col, time_col, feature_cols):
    if df is None:
        return None, None, None, "Upload a combined CSV file first."
    if not sample_col or not label_col:
        return None, None, None, "Choose sample and label columns."
    if not feature_cols:
        return None, None, None, "Choose at least one feature column."

    working = df.copy()
    if time_col and time_col != "(none)":
        working = working.sort_values([sample_col, time_col])
    else:
        working = working.sort_values(sample_col)

    samples = []
    labels = []
    expected_timepoints = None

    for _, group in working.groupby(sample_col, sort=False):
        feature_values = group[feature_cols].apply(pd.to_numeric, errors="coerce")
        if feature_values.isna().any().any():
            return None, None, None, "Feature columns must be numeric and cannot contain blank values."
        if expected_timepoints is None:
            expected_timepoints = len(feature_values)
        elif len(feature_values) != expected_timepoints:
            return None, None, None, "Every sample must have the same number of timepoints."

        samples.append(feature_values.to_numpy(dtype=np.float32))
        labels.append(group[label_col].iloc[0])

    if len(samples) < 2:
        return None, None, None, "At least two samples are required."

    X_raw = torch.tensor(np.stack(samples), dtype=torch.float32)
    return X_raw, np.array(labels), list(feature_cols), None

def parse_sample_csvs(sample_files, labels_df, has_header):
    if not sample_files:
        return None, None, None, "Upload at least two sample CSV files."
    if labels_df is None:
        return None, None, None, "Upload a labels CSV file."

    sorted_files = sorted(sample_files, key=lambda file_obj: file_obj.name)
    labels = labels_df.iloc[:, -1].values
    if len(labels) != len(sorted_files):
        return None, None, None, "The labels CSV must contain exactly one label per sample file."

    samples = []
    feature_names = None
    expected_shape = None
    header = 0 if has_header else None

    for uploaded_file in sorted_files:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=header)
        values = df.apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            return None, None, None, f"{uploaded_file.name} contains non-numeric or blank feature values."
        if expected_shape is None:
            expected_shape = values.shape
            feature_names = [str(col) for col in values.columns]
        elif values.shape != expected_shape:
            return None, None, None, "All sample CSV files must have the same rows x features shape."
        samples.append(values.to_numpy(dtype=np.float32))

    X_raw = torch.tensor(np.stack(samples), dtype=torch.float32)
    return X_raw, labels, feature_names, None

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
        
        # Optional: Add a plot here if you want to visualize the features
        ##with st.expander("Inspect Feature Vector"):
            ##st.bar_chart(st.session_state['snn_features'][0])
             

    # Check if features are ready
    if st.session_state.get('features_ready', False):
        snn_features = st.session_state['snn_features']
        
        st.divider()
        #st.header("Quantum Kernel Classification")
        st.markdown("""
                <h2 style='font-size: 24px;'> Quantum Kernel Classification </h2>
        """, unsafe_allow_html=True)
        
        # 1. Prepare Data
        available_classes = sorted(pd.Series(y).dropna().unique(), key=str)
        default_class_selection = available_classes[:2]
        selected_classes = st.multiselect(
            "Classes for binary quantum classification",
            available_classes,
            default=default_class_selection,
        )

        if len(selected_classes) != 2:
            st.warning("Choose exactly two classes to run the current 2-feature quantum kernel classifier.")
            st.stop()

        class_mask = np.isin(y, selected_classes)
        binary_class = snn_features[class_mask]
        y_final = y[class_mask]
        
        # Select Features based on Indices derived from Names
        feature_indices = [int(feat_idx_1), int(feat_idx_2)]
        X_final = binary_class[:, feature_indices]
        
        col1, col2 = st.columns(2)
        col1.write(f"**Quantum Kernel Input Shape:** {X_final.shape}")
        
        # Show Names in UI
        col2.write(f"**Selected Features:** {feat_name_1}, {feat_name_2} ")

        # 2. Visualise Quantum Circuit
        with st.expander("View Quantum Kernel Circuit"):
            try:
                fig, ax = qml.draw_mpl(kernel)(X_final[0], X_final[1])
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Circuit visualization error: {e}")

        # 3. Quantum Kernel Classification
        if st.button("Run Cross-Validation"):
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=int(SEED))
            y_total2, pred_total2 = [], []

            svm = SVC(kernel=kernel_matrix, C=svm_c)

            progress_bar = st.progress(0)
            status_txt = st.empty()

            for i, (train_index, test_index) in enumerate(kf.split(X_final)):
                status_txt.text(f"Running Fold {i+1}/{k_folds}...")
                
                X_train, X_test = X_final[train_index], X_final[test_index]
                y_train, y_test = y_final[train_index], y_final[test_index]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                svm.fit(X_train, y_train)
                pred2 = svm.predict(X_test)

                y_total2.extend(y_test)
                pred_total2.extend(pred2)
                
                progress_bar.progress((i + 1) / k_folds)

            status_txt.text("Validation Complete.")
            
            # --- FINAL METRICS ---
            final_acc = accuracy(y_total2, pred_total2)
            st.success(f"Final Accuracy: {final_acc:.2%}")
            
            # --- CONFUSION MATRIX ---
            st.markdown("##### Confusion Matrix")   # smaller than subheader
            
            cm = confusion_matrix(y_total2, pred_total2)
            unique_labels = sorted(list(set(y_total2)))
            
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
