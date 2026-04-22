"""Build compact public example datasets for SpikeSense Studio.

The source datasets are from the UEA/UCR Time Series Classification Archive.
This script intentionally writes small balanced subsets so the examples stay
fast in Streamlit Cloud.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "example_data" / "product_examples"

DATASETS = [
    {
        "key": "wearable_basic_motions",
        "name": "Wearable motion recognition",
        "category": "Wearable sensor data",
        "source_dataset": "BasicMotions",
        "url": "https://www.timeseriesclassification.com/aeon-toolkit/BasicMotions.zip",
        "train_file": "BasicMotions_TRAIN.ts",
        "parser": "ts",
        "max_per_class": 6,
        "feature_prefix": "watch_axis",
        "description": "Smartwatch accelerometer and gyroscope signals for standing, running, walking, and badminton.",
    },
    {
        "key": "industrial_wafer",
        "name": "Industrial wafer fault screening",
        "category": "Industrial sensor monitoring",
        "source_dataset": "Wafer",
        "url": "https://www.timeseriesclassification.com/aeon-toolkit/Wafer.zip",
        "train_file": "Wafer_TRAIN.txt",
        "parser": "ucr_txt",
        "max_per_class": 12,
        "feature_prefix": "wafer_signal",
        "label_map": {"-1": "fault_pattern", "1": "normal_pattern"},
        "description": "Semiconductor wafer process sensor traces with normal and faulty patterns.",
    },
    {
        "key": "healthcare_ecg200",
        "name": "Healthcare ECG morphology",
        "category": "Healthcare signal screening",
        "source_dataset": "ECG200",
        "url": "https://www.timeseriesclassification.com/aeon-toolkit/ECG200.zip",
        "train_file": "ECG200_TRAIN.txt",
        "parser": "ucr_txt",
        "max_per_class": 12,
        "feature_prefix": "ecg_lead",
        "label_map": {"-1": "ecg_class_negative", "1": "ecg_class_positive"},
        "description": "ECG beat-shape time series with two morphology classes.",
    },
    {
        "key": "iot_robot_surface",
        "name": "IoT robot surface sensing",
        "category": "IoT anomaly/event classification",
        "source_dataset": "SonyAIBORobotSurface1",
        "url": "https://www.timeseriesclassification.com/aeon-toolkit/SonyAIBORobotSurface1.zip",
        "train_file": "SonyAIBORobotSurface1_TRAIN.txt",
        "parser": "ucr_txt",
        "max_per_class": 12,
        "feature_prefix": "robot_sensor",
        "label_map": {"1": "surface_state_a", "2": "surface_state_b"},
        "description": "Embedded robot sensor traces used to classify surface or movement states.",
    },
]


def _download(url: str, target: Path) -> None:
    if target.exists():
        return
    urllib.request.urlretrieve(url, target)


def _balanced_take(rows: list[tuple[str, list[list[float]]]], max_per_class: int):
    buckets: dict[str, list[list[list[float]]]] = defaultdict(list)
    for label, series in rows:
        if len(buckets[label]) < max_per_class:
            buckets[label].append(series)
    selected = []
    for label in sorted(buckets, key=str):
        for series in buckets[label]:
            selected.append((label, series))
    return selected


def _parse_ucr_txt(text: str, label_map: dict[str, str] | None, feature_prefix: str):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        raw_label = str(int(float(parts[0])))
        label = label_map.get(raw_label, raw_label) if label_map else raw_label
        values = [[float(value)] for value in parts[1:]]
        rows.append((label, values))
    feature_names = [feature_prefix]
    return rows, feature_names


def _parse_ts(text: str, feature_prefix: str):
    rows = []
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data or line.startswith("@") or line.startswith("#"):
            continue
        parts = line.split(":")
        label = parts[-1]
        dimensions = [[float(value) for value in dim.split(",") if value] for dim in parts[:-1]]
        timepoints = list(map(list, zip(*dimensions)))
        rows.append((label, timepoints))
    feature_names = [f"{feature_prefix}_{idx + 1}" for idx in range(len(rows[0][1][0]))]
    return rows, feature_names


def _to_combined_csv(rows, feature_names, output_path: Path) -> None:
    records = []
    for sample_idx, (label, timepoints) in enumerate(rows, start=1):
        sample_id = f"sample_{sample_idx:03d}"
        for time_idx, values in enumerate(timepoints):
            record = {"sample_id": sample_id, "time": time_idx, "label": label}
            record.update({name: values[i] for i, name in enumerate(feature_names)})
            records.append(record)
    pd.DataFrame(records).to_csv(output_path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for dataset in DATASETS:
            zip_path = tmp_path / f"{dataset['source_dataset']}.zip"
            _download(dataset["url"], zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                source_text = archive.read(dataset["train_file"]).decode("utf-8")

            if dataset["parser"] == "ts":
                rows, feature_names = _parse_ts(source_text, dataset["feature_prefix"])
            else:
                rows, feature_names = _parse_ucr_txt(
                    source_text,
                    dataset.get("label_map"),
                    dataset["feature_prefix"],
                )

            rows = _balanced_take(rows, dataset["max_per_class"])
            csv_name = f"{dataset['key']}.csv"
            _to_combined_csv(rows, feature_names, OUTPUT_DIR / csv_name)

            label_counts = defaultdict(int)
            for label, _ in rows:
                label_counts[label] += 1

            first_series = rows[0][1]
            manifest.append({
                "key": dataset["key"],
                "name": dataset["name"],
                "category": dataset["category"],
                "description": dataset["description"],
                "source_dataset": dataset["source_dataset"],
                "source_url": dataset["url"],
                "file": csv_name,
                "samples": len(rows),
                "timepoints": len(first_series),
                "features": len(feature_names),
                "class_counts": dict(sorted(label_counts.items())),
            })

    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
