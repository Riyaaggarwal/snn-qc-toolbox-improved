import os

import numpy as np
import pandas as pd
import torch


DEFAULT_FEATURE_NAMES = [
    "AF3*", "F7*", "F3*", "FC5*", "T7*", "P7*", "O1*",
    "O2*", "P8*", "T8*", "FC6*", "F4*", "F8*", "AF4*",
]


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def combined_csv_template():
    return pd.DataFrame(
        [
            {"sample_id": "trial_001", "time": 0, "label": "class_a", "sensor_1": 0.12, "sensor_2": 0.32},
            {"sample_id": "trial_001", "time": 1, "label": "class_a", "sensor_1": 0.18, "sensor_2": 0.29},
            {"sample_id": "trial_002", "time": 0, "label": "class_b", "sensor_1": 0.76, "sensor_2": 0.15},
            {"sample_id": "trial_002", "time": 1, "label": "class_b", "sensor_1": 0.71, "sensor_2": 0.19},
        ]
    )


def sample_csv_template():
    return pd.DataFrame(
        [
            {"sensor_1": 0.12, "sensor_2": 0.32, "sensor_3": 0.44},
            {"sensor_1": 0.18, "sensor_2": 0.29, "sensor_3": 0.49},
            {"sensor_1": 0.21, "sensor_2": 0.35, "sensor_3": 0.52},
        ]
    )


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
        "class_counts": classes.value_counts().sort_index(key=lambda index: index.astype(str)),
    }


def load_builtin_eeg_dataset(base_path="./example_data/wrist_movement_eeg/"):
    """Load the bundled EEG demo dataset as a raw tensor."""
    if not os.path.exists(base_path):
        return None, None, DEFAULT_FEATURE_NAMES, f"Path not found: {base_path}"

    filenames = [f"sam{idx}_eeg.csv" for idx in range(1, 61)]
    dfs = []

    try:
        for filename in filenames:
            full_path = os.path.join(base_path, filename)
            dfs.append(pd.read_csv(full_path, header=None))
    except FileNotFoundError as exc:
        return None, None, DEFAULT_FEATURE_NAMES, f"Missing File: {exc}"

    fulldf = pd.concat(dfs)
    labels_path = os.path.join(base_path, "tar_class_labels.csv")
    labels = pd.read_csv(labels_path, header=None)
    y_all = labels.values.flatten()

    X_raw = torch.tensor(fulldf.values.reshape(60, 128, 14), dtype=torch.float32)
    return X_raw, y_all, DEFAULT_FEATURE_NAMES, None


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
