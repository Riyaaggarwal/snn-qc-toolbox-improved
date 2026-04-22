"""Smoke-test every built-in product example through the app's core pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neucube import Reservoir
from neucube.encoder import Delta
from neucube.qfeatures import extract_features
from neucube.sampler import SpikeCount
from neucube.training import STDP
from snnqc.analysis import feature_projection
from snnqc.data_loader import available_product_examples, load_product_example_dataset


def run_smoke():
    rows = []
    for example in available_product_examples():
        X_raw, y, feature_names, _, err = load_product_example_dataset(example["key"])
        if err:
            raise RuntimeError(f"{example['key']}: {err}")

        X = Delta(threshold=0.8).encode_dataset(X_raw)
        reservoir = Reservoir(cube_shape=(4, 4, 4), inputs=X.shape[2], c=0.4, l=0.169)
        activity = reservoir.simulate(
            X,
            train=True,
            learning_rule=STDP(a_pos=0.004, a_neg=0.003),
            mem_thr=0.1,
            refractory_period=5,
            verbose=False,
        )
        spike_counts = SpikeCount().sample(activity)
        features = extract_features(spike_counts, reservoir.w_in)

        if features.shape != (X.shape[0], X.shape[2]):
            raise AssertionError(f"{example['key']}: bad feature shape {features.shape}")

        for method in ("PCA", "t-SNE"):
            coords, caption = feature_projection(features, method=method, random_state=42)
            if coords.shape != (X.shape[0], 2):
                raise AssertionError(f"{example['key']} {method}: bad coords {coords.shape}")
            if not np.isfinite(coords).all():
                raise AssertionError(f"{example['key']} {method}: non-finite coordinates")
            if not caption:
                raise AssertionError(f"{example['key']} {method}: missing caption")

        counts = pd.Series(y).value_counts()
        folds = min(3, int(counts.min()))
        if folds >= 2 and len(counts) >= 2:
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, features, y, cv=cv, scoring="accuracy")
            accuracy = float(scores.mean())
        else:
            accuracy = float("nan")

        rows.append({
            "key": example["key"],
            "shape": tuple(X_raw.shape),
            "features": tuple(features.shape),
            "classes": sorted(map(str, set(y))),
            "cv_accuracy": accuracy,
        })

    return rows


if __name__ == "__main__":
    for row in run_smoke():
        print(row)
