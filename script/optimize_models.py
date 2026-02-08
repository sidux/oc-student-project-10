#!/usr/bin/env python3
"""
Optimize model files for lower memory usage in Azure Functions.
"""

import os
import pickle

import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def optimize_embeddings():
    """Convert embeddings from float64 to float32 (halves memory)"""
    path = os.path.join(MODELS_DIR, "article_embeddings.pkl")
    print("\n1. Optimizing article_embeddings.pkl...")

    with open(path, "rb") as f:
        embeddings = pickle.load(f)

    print(f"   Original: {embeddings.dtype}, shape {embeddings.shape}")
    print(f"   Original size: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # Convert to float32
    embeddings_opt = embeddings.astype(np.float32)
    print(f"   Optimized: {embeddings_opt.dtype}")
    print(f"   Optimized size: {embeddings_opt.nbytes / 1024 / 1024:.1f} MB")

    # Save
    with open(path, "wb") as f:
        pickle.dump(embeddings_opt, f, protocol=4)

    new_size = os.path.getsize(path) / 1024 / 1024
    print(f"   File size: {new_size:.1f} MB")


def optimize_ratings():
    """Downcast numeric types in ratings CSV"""
    path = os.path.join(MODELS_DIR, "user_article_ratings.csv")
    print("\n2. Optimizing user_article_ratings.csv...")

    df = pd.read_csv(path)
    original_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"   Original memory: {original_mem:.1f} MB")
    print(f"   Columns: {list(df.columns)}")

    # Downcast numeric columns
    for col in df.columns:
        if df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")

    optimized_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"   Optimized memory: {optimized_mem:.1f} MB")

    # Save with compression
    df.to_csv(path, index=False)
    new_size = os.path.getsize(path) / 1024 / 1024
    print(f"   File size: {new_size:.1f} MB")


def optimize_popularity():
    """Optimize popularity CSV"""
    path = os.path.join(MODELS_DIR, "article_popularity.csv")
    print("\n3. Optimizing article_popularity.csv...")

    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")

    df.to_csv(path, index=False)
    print(f"   File size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")


def optimize_metadata():
    """Optimize metadata CSV"""
    path = os.path.join(MODELS_DIR, "articles_metadata.csv")
    print("\n4. Optimizing articles_metadata.csv...")

    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")

    df.to_csv(path, index=False)
    print(f"   File size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")


def main():
    print("=" * 50)
    print("Optimizing models for Azure Functions")
    print("=" * 50)

    optimize_embeddings()
    optimize_ratings()
    optimize_popularity()
    optimize_metadata()

    print("\n" + "=" * 50)
    print("Final sizes:")
    total = 0
    for f in os.listdir(MODELS_DIR):
        if f.endswith((".pkl", ".csv")):
            size = os.path.getsize(os.path.join(MODELS_DIR, f))
            total += size
            print(f"  {f}: {size / 1024 / 1024:.1f} MB")
    print(f"  TOTAL: {total / 1024 / 1024:.1f} MB")
    print("=" * 50)


if __name__ == "__main__":
    main()
