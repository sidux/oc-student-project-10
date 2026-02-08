#!/usr/bin/env python3
"""
Reduce embedding dimensions using PCA to save memory.
"""

import os
import pickle

import numpy as np
from sklearn.decomposition import PCA

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def main():
    path = os.path.join(MODELS_DIR, "article_embeddings.pkl")

    print("Loading embeddings...")
    with open(path, "rb") as f:
        embeddings = pickle.load(f)

    print(f"Original shape: {embeddings.shape}")
    print(f"Original size: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    print(f"Original dtype: {embeddings.dtype}")

    # Reduce to 50 dimensions
    n_components = 50
    print(f"\nReducing to {n_components} dimensions with PCA...")

    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)

    # Convert to float32
    embeddings_reduced = embeddings_reduced.astype(np.float32)

    print(f"Reduced shape: {embeddings_reduced.shape}")
    print(f"Reduced size: {embeddings_reduced.nbytes / 1024 / 1024:.1f} MB")
    print(f"Variance explained: {sum(pca.explained_variance_ratio_) * 100:.1f}%")

    # Save
    print(f"\nSaving to {path}...")
    with open(path, "wb") as f:
        pickle.dump(embeddings_reduced, f, protocol=4)

    file_size = os.path.getsize(path) / 1024 / 1024
    print(f"File size: {file_size:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
