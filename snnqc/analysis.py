import inspect

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def feature_projection(features, method="PCA", random_state=42):
    """Return robust 2D coordinates for feature-space visualisation."""
    scaled = StandardScaler().fit_transform(features)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    method = str(method).upper()

    if np.isclose(float(np.var(scaled, axis=0).sum()), 0.0):
        coords = np.column_stack([np.arange(scaled.shape[0], dtype=float), np.zeros(scaled.shape[0])])
        return coords, "Feature values are nearly constant — projection shown as sample order."

    if method == "PCA":
        if min(scaled.shape) < 2:
            coords = np.column_stack([scaled[:, 0], np.zeros(scaled.shape[0])])
            return coords, "One-dimensional feature space — plotted on a single horizontal axis."

        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(scaled)
        caption = (
            f"Explained variance — PC1: {pca.explained_variance_ratio_[0]:.1%}, "
            f"PC2: {pca.explained_variance_ratio_[1]:.1%}"
        )
        return coords, caption

    perplexity = min(30, max(5, scaled.shape[0] // 4))
    preprocessed = scaled
    if scaled.shape[1] > 50:
        preprocessed = PCA(n_components=50, random_state=random_state).fit_transform(scaled)

    tsne_kwargs = {
        "n_components": 2,
        "random_state": random_state,
        "perplexity": perplexity,
        "init": "pca" if min(preprocessed.shape) >= 2 else "random",
    }
    tsne_iter_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    tsne_kwargs[tsne_iter_key] = 500

    coords = TSNE(**tsne_kwargs).fit_transform(preprocessed)
    return coords, f"t-SNE  ·  perplexity={perplexity}"
