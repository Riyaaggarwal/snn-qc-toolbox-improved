from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).parent
_DEFAULT_DATA_PATH = _HERE.parent / "example_data" / "wrist_movement_eeg"

DEFAULT_FEATURE_NAMES = [
    "AF3*", "F7*", "F3*", "FC5*", "T7*", "P7*", "O1*",
    "O2*", "P8*", "T8*", "FC6*", "F4*", "F8*", "AF4*",
]


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def combined_csv_template():
    return pd.DataFrame([
        {"sample_id": "trial_001", "time": 0, "label": "class_a", "sensor_1": 0.12, "sensor_2": 0.32},
        {"sample_id": "trial_001", "time": 1, "label": "class_a", "sensor_1": 0.18, "sensor_2": 0.29},
        {"sample_id": "trial_002", "time": 0, "label": "class_b", "sensor_1": 0.76, "sensor_2": 0.15},
        {"sample_id": "trial_002", "time": 1, "label": "class_b", "sensor_1": 0.71, "sensor_2": 0.19},
    ])


def sample_csv_template():
    return pd.DataFrame([
        {"sensor_1": 0.12, "sensor_2": 0.32, "sensor_3": 0.44},
        {"sensor_1": 0.18, "sensor_2": 0.29, "sensor_3": 0.49},
        {"sensor_1": 0.21, "sensor_2": 0.35, "sensor_3": 0.52},
    ])


def labels_csv_template():
    return pd.DataFrame({"label": ["class_a", "class_b", "class_a"]})


def dataset_summary(X_raw, y_data, feature_names):
    classes = pd.Series(y_data).dropna()
    return {
        "samples": int(X_raw.shape[0]),
        "timepoints": int(X_raw.shape[1]),
        "features": int(X_raw.shape[2]),
        "labels": int(len(y_data)),
        "feature_names": list(feature_names),
        "class_counts": classes.value_counts().sort_index(key=lambda idx: idx.astype(str)),
    }


def load_builtin_eeg_dataset(base_path=None):
    """Load the bundled EEG demo dataset. Shape is auto-detected from files."""
    data_dir = Path(base_path) if base_path else _DEFAULT_DATA_PATH

    if not data_dir.exists():
        return None, None, DEFAULT_FEATURE_NAMES, f"Example data directory not found: {data_dir}"

    sample_files = sorted(data_dir.glob("sam*_eeg.csv"))
    if not sample_files:
        return None, None, DEFAULT_FEATURE_NAMES, (
            f"No sample files found in {data_dir} (expected sam*_eeg.csv pattern)"
        )

    labels_path = data_dir / "tar_class_labels.csv"
    if not labels_path.exists():
        return None, None, DEFAULT_FEATURE_NAMES, (
            f"Labels file not found: {labels_path}"
        )

    dfs = []
    for filepath in sample_files:
        try:
            dfs.append(pd.read_csv(filepath, header=None))
        except Exception as exc:
            return None, None, DEFAULT_FEATURE_NAMES, f"Error reading {filepath.name}: {exc}"

    try:
        labels = pd.read_csv(labels_path, header=None)
        y_all = labels.values.flatten()
    except Exception as exc:
        return None, None, DEFAULT_FEATURE_NAMES, f"Error reading labels file: {exc}"

    if len(y_all) != len(dfs):
        return None, None, DEFAULT_FEATURE_NAMES, (
            f"Label count ({len(y_all)}) does not match sample count ({len(dfs)})"
        )

    # Auto-detect shape from first file
    first_df = dfs[0]
    n_timepoints, n_features = first_df.shape
    n_samples = len(dfs)

    for i, df in enumerate(dfs[1:], start=2):
        if df.shape != first_df.shape:
            return None, None, DEFAULT_FEATURE_NAMES, (
                f"Sample {i} shape {df.shape} does not match expected {first_df.shape}"
            )

    try:
        full_array = np.concatenate([df.values for df in dfs]).reshape(n_samples, n_timepoints, n_features)
        X_raw = torch.tensor(full_array, dtype=torch.float32)
    except Exception as exc:
        return None, None, DEFAULT_FEATURE_NAMES, f"Error constructing data tensor: {exc}"

    feature_names = DEFAULT_FEATURE_NAMES if n_features == len(DEFAULT_FEATURE_NAMES) else [
        f"Feature {i + 1}" for i in range(n_features)
    ]
    return X_raw, y_all, feature_names, None


def read_uploaded_csv(uploaded_file, has_header):
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, header=0 if has_header else None)


def parse_combined_csv(df, sample_col, label_col, time_col, feature_cols):
    if df is None:
        return None, None, None, "Upload a combined CSV file first."
    if not sample_col or not label_col:
        return None, None, None, "Choose sample and label columns."
    if not feature_cols:
        return None, None, None, "Choose at least one feature column."

    working = df.copy()
    sort_cols = [sample_col, time_col] if time_col and time_col != "(none)" else [sample_col]
    working = working.sort_values(sort_cols)

    samples, labels = [], []
    expected_timepoints = None

    for _, group in working.groupby(sample_col, sort=False):
        feature_values = group[feature_cols].apply(pd.to_numeric, errors="coerce")
        if feature_values.isna().any().any():
            return None, None, None, "Feature columns must be numeric and cannot contain blank or non-numeric values."
        if expected_timepoints is None:
            expected_timepoints = len(feature_values)
        elif len(feature_values) != expected_timepoints:
            return None, None, None, (
                f"Every sample must have the same number of timepoints "
                f"(expected {expected_timepoints}, got {len(feature_values)})."
            )
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

    sorted_files = sorted(sample_files, key=lambda f: f.name)
    labels = labels_df.iloc[:, -1].values

    if len(labels) != len(sorted_files):
        return None, None, None, (
            f"Labels CSV has {len(labels)} rows but {len(sorted_files)} sample files were uploaded. "
            "There must be exactly one label per sample file."
        )

    samples, feature_names, expected_shape = [], None, None

    for uploaded_file in sorted_files:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=0 if has_header else None)
        values = df.apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            return None, None, None, (
                f"{uploaded_file.name} contains non-numeric or blank values."
            )
        if expected_shape is None:
            expected_shape = values.shape
            feature_names = [str(col) for col in values.columns]
        elif values.shape != expected_shape:
            return None, None, None, (
                f"{uploaded_file.name} has shape {values.shape}, expected {expected_shape}. "
                "All sample CSVs must have the same rows × features."
            )
        samples.append(values.to_numpy(dtype=np.float32))

    X_raw = torch.tensor(np.stack(samples), dtype=torch.float32)
    return X_raw, labels, feature_names, None
