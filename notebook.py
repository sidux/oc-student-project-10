# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.12
# ---

# %% [markdown]
# # Recommendation System
#
# ---
# **Project Context**
#
# 1. Explore **content-based filtering** (using article embeddings).
# 2. Explore **collaborative filtering** (using the Surprise library).
# 3. Propose a **hybrid** or combined approach if needed.
#
#
# **Table of Contents**
#
# 1. [Introduction and Requirements](#introduction)
# 2. [Data Overview](#data-overview)
#     - 2.1 [Available Files](#available-files)
#     - 2.2 [Loading and Exploring Articles Metadata](#articles-metadata)
#     - 2.3 [Loading and Exploring Clicks Data](#clicks-data)
# 3. [Content-Based Filtering Model](#content-based-model)
#     - 3.1 [Loading Embeddings](#loading-embeddings)
#     - 3.2 [Cosine Similarity Computation](#cosine-similarity)
#     - 3.3 [Recommendation Strategy](#cb-recommendation-strategy)
# 4. [Collaborative Filtering Model](#collaborative-filtering)
#     - 4.1 [Rating Definition (Implicit Feedback)](#rating-definition)
#     - 4.2 [Building & Evaluating a CF Model with Surprise](#building-cf)
#         - 4.2.1 [Matrix Factorization (SVD)](#svd)
#         - 4.2.2 [KNN-Based Approaches](#knn)
#     - 4.3 [Generating Top-N Recommendations](#topn)
# 5. [Hybrid Approach](#hybrid-approach)
#     - 5.1 [Weighted Ensemble Method](#weighted-ensemble)
#     - 5.2 [User-Based Selection Strategy](#user-based-selection)
#     - 5.3 [Evaluation of Hybrid Approach](#hybrid-evaluation)
#

# %% [markdown]
# <a id="introduction"></a>
# ## 1. Introduction and Requirements
#
# MyContent, a young startup, needs to provide each user with **5 recommended articles**. The system must:
#
# - **Scale** to handle new users and articles.
# - Provide relevant recommendations by exploiting both:
#   - **User behavior** (clicks, etc.).
#   - **Content properties** (embeddings, metadata).
# - Be **deployed** in a serverless architecture such as Azure Functions.
#
# We will produce two main recommendation approaches:
# 1. A **content-based** model using article embeddings.
# 2. A **collaborative filtering** model using the Surprise library.
#
# Finally, we'll develop a **hybrid approach** combining the strengths of both methods.

# %%
import os
import pickle
import random
import time

# For ignoring warnings in the notebook (not recommended for production)
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Surprise for collaborative filtering
from surprise import SVD, Dataset, KNNWithMeans, Reader, accuracy
from surprise.model_selection import train_test_split

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

# Result directories for saving plots and reports
RESULT_DIR = Path("result")
PLOTS_DIR = RESULT_DIR / "plots"
REPORTS_DIR = RESULT_DIR / "reports"

for directory in (RESULT_DIR, PLOTS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def save_plot(fig_name: str, fig=None):
    """Save current or provided figure to PLOTS_DIR."""
    fig_path = PLOTS_DIR / fig_name
    if fig is not None:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    else:
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")


def save_report(content: str, report_name: str):
    """Save text content to REPORTS_DIR."""
    report_path = REPORTS_DIR / report_name
    with open(report_path, "w") as f:
        f.write(content)
    print(f"Saved: {report_path}")


# %% [markdown]
# <a id="data-overview"></a>
# ## 2. Data Overview
# The **Kaggle** dataset includes:
#
# - `articles_metadata.csv`
# - `articles_embeddings.pickle`
# - `clicks_sample.csv` (plus multiple hourly files in `clicks/` folder)
#
# For demonstration, we can load the sample clicks and metadata. If needed, we can merge or filter data for faster processing.

# %% [markdown]
# <a id="available-files"></a>
# ### 2.1 Available Files
#
# 1. **`articles_metadata.csv`**: Basic article metadata
#    - `article_id`
#    - `category_id`
#    - `created_at_ts`
#    - `publisher_id`
#    - `words_count`
#
# 2. **`articles_embeddings.pickle`**: Article embeddings (NumPy array or matrix). Each article has a 250-dimensional vector representing its content.
# 3. **`clicks_sample.csv`**: Contains user interactions (clicks) with articles for a subset of sessions. Columns include:
#    - `user_id`, `session_id`, `session_start`, `session_size`
#    - `click_article_id`
#    - `click_timestamp`
#    - Various user environment and location columns.
#
# 4. **`clicks/`**: Additional hourly data files with the same schema as `clicks_sample.csv`.

# %% [markdown]
# <a id="articles-metadata"></a>
# ### 2.2 Loading and Exploring Articles Metadata

# %%
PATH_ARTICLES_DATA = "data/articles_metadata.csv"

articles_df = pd.read_csv(PATH_ARTICLES_DATA)
print("articles_df shape:", articles_df.shape)
articles_df.head()

# %% [markdown]
# Let's quickly inspect for missing values and distribution:

# %%
articles_df.info()

# %%
# Check if any value is missing
articles_df.isna().sum()

# %%
# Distribution of words_count
fig, ax = plt.subplots(figsize=(12, 8))
sns.histplot(articles_df["words_count"], ax=ax)
plt.xlim([0, 400])
plt.title("Distribution of Words per Article")
plt.xlabel("Words Count")
plt.ylabel("Frequency")
save_plot("01_words_distribution.png")
plt.show()

# %%
# Convert timestamp to datetime for better analysis
articles_df["created_at"] = pd.to_datetime(articles_df["created_at_ts"], unit="ms")

# Plot article publication over time
plt.figure(figsize=(12, 6))
articles_df["created_at"].dt.date.value_counts().sort_index().plot()
plt.title("Number of Articles Published Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.tight_layout()
save_plot("02_articles_over_time.png")
plt.show()

# %%
# Inspect categories distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(
    data=articles_df,  # sampling just for a faster plot
    x="category_id",
    color="deepskyblue",
    edgecolor="black",
    ax=ax,
)
plt.title("Category Distribution")
plt.xlabel("Category ID")
plt.ylabel("Count")
plt.xticks(rotation=45)
save_plot("03_category_distribution.png")
plt.show()

# %% [markdown]
# <a id="clicks-data"></a>
# ### 2.3 Loading and Exploring Clicks Data
#
# Below we first show how to load the **sample** file for reference. However, in order to use **all** the hourly click files, we will merge them into a single DataFrame.

# %%
# (1) Sample clicks file (optional reference)
PATH_CLICK_SAMPLE_DATA = "data/clicks_sample.csv"
clicks_samp_df = pd.read_csv(PATH_CLICK_SAMPLE_DATA)
print("clicks_samp_df shape:", clicks_samp_df.shape)
clicks_samp_df.head()

# %%
clicks_samp_df.info()

# %%
print("Unique users in sample:", clicks_samp_df["user_id"].nunique())
print("Unique articles in sample:", clicks_samp_df["click_article_id"].nunique())
print("Unique sessions in sample:", clicks_samp_df["session_id"].nunique())

# Verify session sizes
session_sizes = clicks_samp_df.groupby("session_id").agg({"session_size": "first", "user_id": "count"}).rename(columns={"user_id": "actual_session_size"})
print("\nSession size verification (first 5 sessions):")
print(session_sizes.head())

# Check if actual clicks match the declared session_size
mismatched_sessions = session_sizes[session_sizes["session_size"] != session_sizes["actual_session_size"]]
print(f"\nNumber of sessions with mismatched sizes: {len(mismatched_sessions)}")
if len(mismatched_sessions) > 0:
    print("Sample of mismatched sessions:")
    print(mismatched_sessions.head())

# %% [markdown]
#
# **We will now load all `clicks_hour_*.csv` files** from the `clicks/` folder to create a larger dataset.

# %%
CLICKS_FOLDER = "data/clicks/"
all_click_files = [os.path.join(CLICKS_FOLDER, f) for f in os.listdir(CLICKS_FOLDER) if f.endswith(".csv")]
all_click_files.sort()

# Concatenate all into one DataFrame
list_clicks_full = []
for path in all_click_files:
    df_temp = pd.read_csv(path)
    list_clicks_full.append(df_temp)

df_clicks_full = pd.concat(list_clicks_full, ignore_index=True)
print("Merged clicks shape:", df_clicks_full.shape)
df_clicks_full.head()

# %%
n_users = df_clicks_full["user_id"].nunique()
n_articles_clicks = df_clicks_full["click_article_id"].nunique()
n_sessions = df_clicks_full["session_id"].nunique()
n_interactions = len(df_clicks_full)

print("Unique users in ALL clicks:", n_users)
print("Unique articles in ALL clicks:", n_articles_clicks)
print("Unique sessions in ALL clicks:", n_sessions)

# Save data overview report
data_report = f"""Data Overview Report
====================
Articles Dataset:
  - Total articles: {len(articles_df):,}
  - Embedding dimensions: 250

Clicks Dataset:
  - Total interactions: {n_interactions:,}
  - Unique users: {n_users:,}
  - Unique articles clicked: {n_articles_clicks:,}
  - Unique sessions: {n_sessions:,}
  - Avg clicks per user: {n_interactions / n_users:.2f}
  - Matrix sparsity: {(1 - n_interactions / (n_users * n_articles_clicks)) * 100:.4f}%
"""
save_report(data_report, "data_overview.txt")

# Convert timestamp to datetime
df_clicks_full["click_datetime"] = pd.to_datetime(df_clicks_full["click_timestamp"], unit="ms")

# Session analysis - check distribution of session sizes
session_stats = (
    df_clicks_full.groupby("session_id")
    .agg({"user_id": "first", "session_size": "first", "click_article_id": "count"})
    .rename(columns={"click_article_id": "actual_clicks"})
)

plt.figure(figsize=(10, 6))
sns.histplot(data=session_stats, x="actual_clicks", bins=30)
plt.title("Distribution of Clicks per Session")
plt.xlabel("Number of Clicks")
plt.ylabel("Count")
plt.tight_layout()
save_plot("04_clicks_per_session.png")
plt.show()

# Look at sessions with high click counts
high_activity_sessions = session_stats[session_stats["actual_clicks"] > 20]
print(f"\nNumber of high-activity sessions (>20 clicks): {len(high_activity_sessions)}")

# %% [markdown]
# ## 3. Content-Based Filtering Model
#
# Content-based filtering focuses on **similarities between items** (in this case, articles). If we have vector embeddings for each article (derived from text and/or metadata), we can compute a **cosine similarity** matrix between articles to find the most similar articles to a given one.
#
# In practice, for each user, we can:
# 1. Identify the article(s) the user has recently clicked.
# 2. For each such article, compute the **top-5** most similar articles.
# 3. Aggregate or combine these top articles and remove duplicates (and items the user already has seen).
#
# **Note**: In production, computing full pairwise similarities can be expensive for very large sets. Techniques like approximate nearest neighbors or dimension reduction (PCA) might be applied to speed up computations.

# %% [markdown]
# <a id="loading-embeddings"></a>
# ### 3.1 Loading Embeddings
#
# A file `articles_embeddings.pickle` is provided. Each row corresponds to `article_id`, each column dimension is part of the 250-dimensional embedding vector.

# %%
PATH_ARTICLES_EMBED = "data/articles_embeddings.pickle"
with open(PATH_ARTICLES_EMBED, "rb") as f:
    articles_embeddings_raw = pickle.load(f)

print("Shape of raw articles_embeddings:", articles_embeddings_raw.shape)
print("Data type:", articles_embeddings_raw.dtype)
print("Memory usage (raw):", articles_embeddings_raw.nbytes / 1024 / 1024, "MB")
print("Any NaN values in embeddings:", np.isnan(articles_embeddings_raw).any())

# %% [markdown]
# ### 3.1.1 Dimensionality Reduction Study with PCA
#
# The original embeddings are **250-dimensional**, leading to:
# - High memory usage (~347 MB for 364K articles)
# - Slower similarity computations
# - Deployment constraints (Azure Functions: 1.5 GB limit)
#
# We'll conduct a proper **PCA analysis** to determine the optimal number of components.

# %% [markdown]
# #### Step 1: Full PCA Analysis
#
# First, we fit PCA with all components to analyze the variance distribution.

# %%
from sklearn.decomposition import PCA

# Fit PCA with all components to study variance distribution
pca_full = PCA(random_state=42)
pca_full.fit(articles_embeddings_raw)

# Analyze explained variance
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("=== PCA Variance Analysis ===")
print(f"Total components: {len(explained_variance)}")
print("\nTop 10 components explained variance:")
for i in range(10):
    print(f"  PC{i + 1}: {explained_variance[i] * 100:.2f}% (cumulative: {cumulative_variance[i] * 100:.2f}%)")

# %% [markdown]
# #### Step 2: Scree Plot and Cumulative Variance
#
# The **scree plot** helps identify the "elbow" where adding more components yields diminishing returns.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot (first 50 components for clarity)
ax1 = axes[0]
components_to_plot = 50
ax1.bar(range(1, components_to_plot + 1), explained_variance[:components_to_plot] * 100, alpha=0.7, color="steelblue", label="Individual")
ax1.plot(range(1, components_to_plot + 1), cumulative_variance[:components_to_plot] * 100, "ro-", linewidth=2, markersize=4, label="Cumulative")
ax1.axhline(y=95, color="green", linestyle="--", alpha=0.7, label="95% threshold")
ax1.axhline(y=90, color="orange", linestyle="--", alpha=0.7, label="90% threshold")
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance (%)")
ax1.set_title("Scree Plot - First 50 Components")
ax1.legend(loc="center right")
ax1.grid(True, alpha=0.3)

# Full cumulative variance plot
ax2 = axes[1]
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, "b-", linewidth=2)
ax2.axhline(y=95, color="green", linestyle="--", alpha=0.7, label="95% variance")
ax2.axhline(y=99, color="red", linestyle="--", alpha=0.7, label="99% variance")

# Mark key thresholds
for threshold in [0.90, 0.95, 0.99]:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    ax2.axvline(x=n_components, color="gray", linestyle=":", alpha=0.5)
    ax2.annotate(
        f"{int(threshold * 100)}%: {n_components} comp.", xy=(n_components, threshold * 100), xytext=(n_components + 10, threshold * 100 - 3), fontsize=9
    )

ax2.set_xlabel("Number of Components")
ax2.set_ylabel("Cumulative Explained Variance (%)")
ax2.set_title("Cumulative Explained Variance")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_plot("05_pca_explained_variance.png")
plt.show()

# %% [markdown]
# #### Step 3: Determine Optimal Number of Components
#
# We consider multiple criteria:
# 1. **Variance threshold**: 90%, 95%, 99%
# 2. **Memory constraints**: Azure Functions 1.5GB limit
# 3. **Recommendation quality**: Impact on similarity calculations

# %%
# Find components needed for different variance thresholds
thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
print("=== Components Required for Variance Thresholds ===\n")
print(f"{'Threshold':<12} {'Components':<12} {'Memory (MB)':<15} {'Reduction':<12}")
print("-" * 55)

for threshold in thresholds:
    n_comp = np.argmax(cumulative_variance >= threshold) + 1
    memory_mb = (articles_embeddings_raw.shape[0] * n_comp * 4) / (1024 * 1024)  # float32
    reduction = (1 - n_comp / 250) * 100
    print(f"{threshold * 100:.0f}%{'':<9} {n_comp:<12} {memory_mb:<15.1f} {reduction:.0f}%")

# %% [markdown]
# #### Step 4: Quality Impact Analysis
#
# We evaluate how dimensionality reduction affects **cosine similarity** accuracy
# by comparing similarities in original vs reduced space.

# %%
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_pca_quality(embeddings_original, n_components_list, n_samples=1000):
    """
    Evaluate how well PCA preserves pairwise similarities.
    """
    # Sample random pairs for evaluation
    np.random.seed(42)
    n_articles = embeddings_original.shape[0]
    idx1 = np.random.choice(n_articles, n_samples, replace=False)
    idx2 = np.random.choice(n_articles, n_samples, replace=False)

    # Original similarities
    original_sims = np.array([cosine_similarity(embeddings_original[i1 : i1 + 1], embeddings_original[i2 : i2 + 1])[0, 0] for i1, i2 in zip(idx1, idx2)])

    results = []
    for n_comp in n_components_list:
        pca = PCA(n_components=n_comp, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings_original)

        # Reduced similarities
        reduced_sims = np.array([cosine_similarity(embeddings_reduced[i1 : i1 + 1], embeddings_reduced[i2 : i2 + 1])[0, 0] for i1, i2 in zip(idx1, idx2)])

        # Correlation between original and reduced similarities
        correlation = np.corrcoef(original_sims, reduced_sims)[0, 1]
        mae = np.mean(np.abs(original_sims - reduced_sims))
        variance_retained = sum(pca.explained_variance_ratio_) * 100

        results.append({"n_components": n_comp, "variance_retained": variance_retained, "similarity_correlation": correlation, "similarity_mae": mae})

    return pd.DataFrame(results)


# Evaluate different component counts
component_options = [10, 20, 30, 40, 50, 75, 100, 150, 200]
quality_results = evaluate_pca_quality(articles_embeddings_raw, component_options)

print("=== PCA Quality Impact on Cosine Similarity ===\n")
print(quality_results.to_string(index=False))

# %%
# Visualize quality metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Similarity correlation
ax1 = axes[0]
ax1.plot(quality_results["n_components"], quality_results["similarity_correlation"], "bo-", linewidth=2, markersize=8)
ax1.axhline(y=0.99, color="green", linestyle="--", alpha=0.7, label="99% correlation")
ax1.axhline(y=0.95, color="orange", linestyle="--", alpha=0.7, label="95% correlation")
ax1.set_xlabel("Number of Components")
ax1.set_ylabel("Similarity Correlation with Original")
ax1.set_title("Preservation of Cosine Similarities")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.9, 1.01)

# Memory vs Quality trade-off
ax2 = axes[1]
memory_mb = [(articles_embeddings_raw.shape[0] * n * 4) / (1024 * 1024) for n in quality_results["n_components"]]
ax2.scatter(memory_mb, quality_results["similarity_correlation"], c=quality_results["n_components"], cmap="viridis", s=100)
for i, row in quality_results.iterrows():
    mem = (articles_embeddings_raw.shape[0] * row["n_components"] * 4) / (1024 * 1024)
    ax2.annotate(f"n={int(row['n_components'])}", (mem, row["similarity_correlation"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax2.axvline(x=100, color="red", linestyle="--", alpha=0.7, label="100 MB target")
ax2.set_xlabel("Memory Usage (MB)")
ax2.set_ylabel("Similarity Correlation")
ax2.set_title("Memory vs Quality Trade-off")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_plot("06_memory_quality_tradeoff.png")
plt.show()

# %% [markdown]
# #### Step 5: Final Selection
#
# Based on our analysis:
#
# | Criteria | Requirement | Selected (n=50) |
# |----------|-------------|-----------------|
# | Variance retained | > 90% | **~95%** ✓ |
# | Similarity correlation | > 0.95 | **~0.99** ✓ |
# | Memory usage | < 100 MB | **~69 MB** ✓ |
# | Azure Functions compatible | < 1.5 GB total | **Yes** ✓ |
#
# **Decision: n_components = 50** provides the best balance between quality and memory efficiency.

# %%
# Apply final PCA transformation
PCA_N_COMPONENTS = 50  # Selected based on analysis above

print(f"\n{'=' * 50}")
print(f"FINAL PCA TRANSFORMATION: {articles_embeddings_raw.shape[1]} → {PCA_N_COMPONENTS} dimensions")
print(f"{'=' * 50}")

pca_final = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
articles_embeddings = pca_final.fit_transform(articles_embeddings_raw).astype(np.float32)

variance_retained = sum(pca_final.explained_variance_ratio_) * 100
memory_before = articles_embeddings_raw.nbytes / 1024 / 1024
memory_after = articles_embeddings.nbytes / 1024 / 1024

print("\nResults:")
print(f"  Shape: {articles_embeddings_raw.shape} → {articles_embeddings.shape}")
print(f"  Memory: {memory_before:.1f} MB → {memory_after:.1f} MB ({(1 - memory_after / memory_before) * 100:.0f}% reduction)")
print(f"  Variance retained: {variance_retained:.1f}%")
print(f"  Data type: {articles_embeddings.dtype}")

# Free raw embeddings to save memory
del articles_embeddings_raw

# %% [markdown]
# **Conclusion**: PCA reduction from 250 to 50 dimensions achieves:
# - **80% memory reduction** (347 MB → 69 MB)
# - **95% variance retained**
# - **99% similarity correlation** with original embeddings
# - Enables deployment on Azure Functions Consumption Plan

# %% [markdown]
# <a id="cosine-similarity"></a>
# ### 3.2 Cosine Similarity Computation
#
# Cosine similarity between two vectors `u` and `v` is (u · v) / (||u|| ||v||).
# We can compute pairwise similarities with `sklearn.metrics.pairwise.cosine_similarity`, or do it on the fly for only the user's target articles.

# %%


def get_top_n_similar_articles(article_id, embeddings_matrix, top_n=5):
    """
    Given the article_id and the embeddings matrix,
    returns a list of top_n most similar article IDs (excluding itself).
    """
    # Reshape the target article's embedding into a 2D array
    target_embed = embeddings_matrix[article_id].reshape(1, -1)

    # Compute similarity with all articles
    similarities = cosine_similarity(target_embed, embeddings_matrix)[0]  # 1D array
    # The similarity with itself is the highest, so we exclude it
    similarities[article_id] = -1  # So it won't appear in the top

    # Get top_n indices sorted by descending similarity
    top_indices = np.argsort(similarities)[::-1][:top_n]

    return top_indices, similarities[top_indices]


# Example usage (with a random article_id)
example_article_id = 12345  # choose an ID that exists in 0..(num_articles-1)
similar_articles, sim_scores = get_top_n_similar_articles(example_article_id, articles_embeddings, top_n=5)

print("Target Article ID:", example_article_id)
print("Top 5 similar articles:", similar_articles)
print("Similarity scores:", sim_scores)


# %% [markdown]
# <a id="cb-recommendation-strategy"></a>
# ### 3.3 Content-Based Recommendation Strategy
#
# For a user who has clicked multiple articles, we can adopt **one** of these strategies:
# 1. Take the **most recent** clicked article and retrieve the top similar articles.
# 2. Compute an **average embedding** of all recently clicked articles and find the top neighbors.
# 3. Take multiple articles and combine the top neighbors for each.
#
# Let's implement all three approaches and compare them:


# %%
def get_content_recommendations_recent(user_id, clicks_df, embeddings_matrix, top_n=5):
    """
    Get recommendations based on the most recent article clicked by the user.

    Args:
        user_id: ID of the user
        clicks_df: DataFrame with click data
        embeddings_matrix: Matrix of article embeddings
        top_n: Number of recommendations to return

    Returns:
        List of recommended article IDs
    """
    # Get user's clicks, sorted by timestamp (most recent first)
    user_clicks = clicks_df[clicks_df["user_id"] == user_id].sort_values(by="click_timestamp", ascending=False)

    if len(user_clicks) == 0:
        return []  # No clicks for this user

    # Get most recent article
    most_recent_article = user_clicks.iloc[0]["click_article_id"]

    # Find similar articles
    similar_articles, _ = get_top_n_similar_articles(most_recent_article, embeddings_matrix, top_n=top_n)

    # Filter out articles the user has already clicked
    clicked_articles = set(user_clicks["click_article_id"])
    recommendations = [art for art in similar_articles if art not in clicked_articles]

    return recommendations[:top_n]


def get_content_recommendations_average(user_id, clicks_df, embeddings_matrix, top_n=5, max_history=10):
    """
    Get recommendations based on the average embedding of user's recently clicked articles.

    Args:
        user_id: ID of the user
        clicks_df: DataFrame with click data
        embeddings_matrix: Matrix of article embeddings
        top_n: Number of recommendations to return
        max_history: Maximum number of recent articles to consider

    Returns:
        List of recommended article IDs
    """
    # Get user's clicks, sorted by timestamp (most recent first)
    user_clicks = clicks_df[clicks_df["user_id"] == user_id].sort_values(by="click_timestamp", ascending=False)

    if len(user_clicks) == 0:
        return []  # No clicks for this user

    # Get most recent articles (up to max_history)
    recent_articles = user_clicks.head(max_history)["click_article_id"].values

    # Calculate average embedding
    avg_embedding = np.mean([embeddings_matrix[art_id] for art_id in recent_articles], axis=0)

    # Reshape to 2D array for cosine_similarity
    avg_embedding = avg_embedding.reshape(1, -1)

    # Compute similarity with all articles
    similarities = cosine_similarity(avg_embedding, embeddings_matrix)[0]

    # Set similarity of clicked articles to -1 so they won't be recommended
    clicked_articles = set(user_clicks["click_article_id"])
    for art_id in clicked_articles:
        similarities[art_id] = -1

    # Get top_n indices
    recommendations = np.argsort(similarities)[::-1][:top_n]

    return recommendations


def get_content_recommendations_combined(user_id, clicks_df, embeddings_matrix, top_n=5, max_history=3, per_article=3):
    """
    Get recommendations by combining top similar articles from multiple recently clicked articles.

    Args:
        user_id: ID of the user
        clicks_df: DataFrame with click data
        embeddings_matrix: Matrix of article embeddings
        top_n: Number of recommendations to return
        max_history: Maximum number of recent articles to consider
        per_article: Number of similar articles to get for each clicked article

    Returns:
        List of recommended article IDs
    """
    # Get user's clicks, sorted by timestamp (most recent first)
    user_clicks = clicks_df[clicks_df["user_id"] == user_id].sort_values(by="click_timestamp", ascending=False)

    if len(user_clicks) == 0:
        return []  # No clicks for this user

    # Get most recent articles (up to max_history)
    recent_articles = user_clicks.head(max_history)["click_article_id"].values

    # Get recommendations for each recent article
    all_recommendations = []
    all_similarities = []

    for art_id in recent_articles:
        similar_ids, sim_scores = get_top_n_similar_articles(art_id, embeddings_matrix, top_n=per_article)
        all_recommendations.extend(similar_ids)
        all_similarities.extend(sim_scores)

    # Create a dictionary of article_id -> similarity score
    # If an article appears multiple times, keep the highest score
    recommendations_dict = {}
    for art_id, score in zip(all_recommendations, all_similarities):
        if art_id not in recommendations_dict or score > recommendations_dict[art_id]:
            recommendations_dict[art_id] = score

    # Remove articles the user has already clicked
    clicked_articles = set(user_clicks["click_article_id"])
    for art_id in clicked_articles:
        if art_id in recommendations_dict:
            del recommendations_dict[art_id]

    # Sort by similarity score and get top_n
    recommendations = sorted(recommendations_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [art_id for art_id, _ in recommendations]


# Example usage
test_user_id = 23  # Replace with a user ID that exists in your dataset
print("Content-based recommendations using most recent article:")
cb_recent = get_content_recommendations_recent(test_user_id, df_clicks_full, articles_embeddings)
print(cb_recent)

print("\nContent-based recommendations using average embedding:")
cb_avg = get_content_recommendations_average(test_user_id, df_clicks_full, articles_embeddings)
print(cb_avg)

print("\nContent-based recommendations using combined approach:")
cb_combined = get_content_recommendations_combined(test_user_id, df_clicks_full, articles_embeddings)
print(cb_combined)

# %% [markdown]
# ## 4. Collaborative Filtering Model
#
# **Collaborative Filtering** (CF) makes recommendations based on how **similar users** or **similar items** relate to each other. If two users share similar interactions, they might enjoy the same items.
#
# We do **not** have explicit user ratings. Instead, we have **implicit feedback** (user clicks). A typical method is to:
# 1. Construct an **implicit rating** from click frequency or session-based metrics.
# 2. Use a CF algorithm (e.g., **`Surprise`** library) to learn user–item preferences.

# %% [markdown]
# <a id="rating-definition"></a>
# ### 4.1 Defining Ratings from Clicks
#
# Instead of just using raw click counts, we'll create a more sophisticated rating scheme that:
# 1. Accounts for multiple clicks within the same session
# 2. Gives higher weight to more recent interactions
# 3. Normalizes ratings to prevent users with many clicks from dominating

# %%
# Calculate days since the most recent click (for recency weighting)
max_timestamp = df_clicks_full["click_timestamp"].max()
df_clicks_full["days_since_click"] = (max_timestamp - df_clicks_full["click_timestamp"]) / (24 * 60 * 60 * 1000)

# Ensure numeric dtype (fixes potential object dtype issues)
df_clicks_full["days_since_click"] = pd.to_numeric(df_clicks_full["days_since_click"], errors="coerce").fillna(0)

# Calculate recency factor (exponential decay)
df_clicks_full["recency_factor"] = np.exp(-0.1 * df_clicks_full["days_since_click"].values)

# %% [markdown]
# #### Vectorized Rating Calculation
#
# Instead of using slow `groupby().apply()`, we use **vectorized aggregations**:

# %%
print("Calculating user-article ratings (vectorized)...")
start_time = time.time()

# Step 1: Get unique sessions per (user, article) with their session_size
session_info = df_clicks_full.groupby(["user_id", "click_article_id", "session_id"])["session_size"].first().reset_index()

# Step 2: Aggregate by (user, article)
# - total_session_size: sum of unique session sizes
# - num_sessions: count of unique sessions
# - recorded_clicks: number of click records
session_agg = (
    session_info.groupby(["user_id", "click_article_id"]).agg(total_session_size=("session_size", "sum"), num_sessions=("session_id", "count")).reset_index()
)

click_agg = (
    df_clicks_full.groupby(["user_id", "click_article_id"])
    .agg(recorded_clicks=("click_timestamp", "count"), mean_recency=("recency_factor", "mean"))
    .reset_index()
)

# Step 3: Merge aggregations
user_article_ratings = session_agg.merge(click_agg, on=["user_id", "click_article_id"])

# Step 4: Calculate rating components (vectorized)
# Extract as numpy arrays with explicit float64 dtype to avoid ufunc errors
total_session_size = user_article_ratings["total_session_size"].to_numpy(dtype=np.float64)
num_sessions = user_article_ratings["num_sessions"].to_numpy(dtype=np.float64)
recorded_clicks = user_article_ratings["recorded_clicks"].to_numpy(dtype=np.float64)
mean_recency = user_article_ratings["mean_recency"].to_numpy(dtype=np.float64)

# Adjusted clicks = max(total_session_size, recorded_clicks)
adjusted_clicks = np.maximum(total_session_size, recorded_clicks)

# Session diversity = num_sessions / adjusted_clicks
session_diversity = num_sessions / np.clip(adjusted_clicks, 1, None)

# Base rating (logarithmic scale)
base_rating = 1.0 + np.log1p(adjusted_clicks)

# Session bonus (1 to 2 multiplier)
session_bonus = 1.0 + session_diversity

# Recency bonus (1 to 2 multiplier)
recency_bonus = 1.0 + mean_recency

# Final rating
rating = base_rating * session_bonus * recency_bonus

# Assign back to dataframe
user_article_ratings["rating"] = rating

# Keep only needed columns
user_article_ratings = user_article_ratings[["user_id", "click_article_id", "rating"]]

elapsed = time.time() - start_time
print(f"Rating calculation completed in {elapsed:.1f} seconds")
print(f"Generated {len(user_article_ratings):,} user-article ratings")

# Explore the distribution of calculated ratings
plt.figure(figsize=(10, 6))
sns.histplot(data=user_article_ratings, x="rating", bins=30)
plt.title("Distribution of Calculated Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
save_plot("07_ratings_distribution.png")
plt.show()

# Show some example ratings
print("Sample of user-article ratings:")
print(user_article_ratings.sort_values(by="rating", ascending=False).head(10))

# %% [markdown]
# <a id="building-cf"></a>
# ### 4.2 Building & Evaluating a CF Model with Surprise
#
# We can use the [Surprise library](https://surprise.readthedocs.io/) to:
# 1. Load our (user, item, rating) data.
# 2. Train a collaborative filtering model.
# 3. Evaluate it on a held-out test set (RMSE, MAE, etc.).

# %%
# Build the full Surprise dataset
min_rating = user_article_ratings["rating"].min()
max_rating = user_article_ratings["rating"].max()
reader = Reader(rating_scale=(min_rating, max_rating))

# Build the full Surprise dataset
data = Dataset.load_from_df(user_article_ratings[["user_id", "click_article_id", "rating"]], reader)

# Split into train/test
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

print("Trainset size (surprise internal) ~", trainset.n_ratings)
print("Testset size =", len(testset))

print("Trainset size (surprise internal) ~", trainset.n_ratings)
print("Testset size =", len(testset))

# %% [markdown]
# <a id="svd"></a>
# #### 4.2.1 Matrix Factorization with SVD
#
# **SVD** is a popular matrix factorization algorithm for CF. Let's tune its hyperparameters.

# %%
# Define a grid of hyperparameters to search
# best hyper params found  50  30    0.01     0.02
param_grid = {"n_factors": [50], "n_epochs": [30], "lr_all": [0.01], "reg_all": [0.02]}

# Test different hyperparameter combinations
results = []
for n_factors in param_grid["n_factors"]:
    for n_epochs in param_grid["n_epochs"]:
        for lr_all in param_grid["lr_all"]:
            for reg_all in param_grid["reg_all"]:
                svd = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, random_state=42)

                # Train and evaluate
                svd.fit(trainset)
                predictions = svd.test(testset)
                rmse = accuracy.rmse(predictions, verbose=False)
                mae = accuracy.mae(predictions, verbose=False)

                results.append({"n_factors": n_factors, "n_epochs": n_epochs, "lr_all": lr_all, "reg_all": reg_all, "rmse": rmse, "mae": mae})

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)
print("Best hyperparameters (by RMSE):")
print(results_df.sort_values(by="rmse").head(1))

# Train the best model
best_params = results_df.sort_values(by="rmse").iloc[0]
svd_model = SVD(
    n_factors=int(best_params["n_factors"]),
    n_epochs=int(best_params["n_epochs"]),
    lr_all=best_params["lr_all"],
    reg_all=best_params["reg_all"],
    random_state=42,
)
svd_model.fit(trainset)

# Evaluate the best model
predictions_svd = svd_model.test(testset)
rmse_svd = accuracy.rmse(predictions_svd, verbose=True)
mae_svd = accuracy.mae(predictions_svd, verbose=True)

# Save SVD model performance report
svd_report = f"""SVD Model Performance Report
============================
Best Hyperparameters:
  - n_factors: {int(best_params["n_factors"])}
  - n_epochs: {int(best_params["n_epochs"])}
  - learning_rate: {best_params["lr_all"]}
  - regularization: {best_params["reg_all"]}

Evaluation Metrics:
  - RMSE: {rmse_svd:.4f}
  - MAE: {mae_svd:.4f}

Training Set Size: {trainset.n_ratings}
Test Set Size: {len(testset)}
"""
save_report(svd_report, "svd_model_performance.txt")


# %% [markdown]
# <a id="knn"></a>
# #### 4.2.2 KNN-Based Models
#
# We can also try an **item-based** KNN approach (KNNWithMeans) using cosine similarity:

# %% [markdown]
# *Pearson similarity* (also called Pearson correlation coefficient) measures the linear relationship between two sets of values. It tells us how well one variable predicts another rather than just their absolute similarity
# Unlike cosine similarity, which measures the angle between vectors, Pearson similarity looks at how much two vectors co-vary (increase or decrease together) while ignoring their magnitudes.
# Resource-friendly approach to KNN parameter tuning
# This version reduces memory usage and CPU load
#
# Try different similarity metrics and k values
# Reduced parameter space - focus on most promising options first
# sim_options_list = [
#     {'name': 'cosine', 'user_based': False},  # item-based cosine (often works best)
#     {'name': 'pearson', 'user_based': False}  # item-based pearson
# ]
#
# k_values = [40]  # Start with just one k value for initial testing
#
# Create much smaller test set for faster evaluation
# random.seed(42)
# sample_size = min(2000, len(testset))  # Much smaller sample (2000 instead of 10000)
# sampled_testset = random.sample(testset, sample_size) if len(testset) > sample_size else testset
# print(f"Using {len(sampled_testset)} test samples for initial evaluation")
#
# Sequential approach - evaluate one config at a time
# print("Testing KNN configurations sequentially to avoid high memory usage...")
# knn_results = []
#
# total_configs = len(sim_options_list) * len(k_values)
# current_config = 0
#
# for sim_options in sim_options_list:
#     for k in k_values:
#         current_config += 1
#         print(f"Testing configuration {current_config}/{total_configs}: {sim_options}, k={k}")
#
#         start_time = time.time()
#
#         try:
#             # Initialize and train KNN model
#             knn = KNNWithMeans(k=k, sim_options=sim_options, verbose=False)
#             knn.fit(trainset)
#
#             # Test on sample
#             predictions = knn.test(sampled_testset)
#             rmse = accuracy.rmse(predictions, verbose=False)
#             mae = accuracy.mae(predictions, verbose=False)
#
#             # Record results
#             knn_results.append({
#                 'sim_name': sim_options['name'],
#                 'user_based': sim_options['user_based'],
#                 'k': k,
#                 'rmse': rmse,
#                 'mae': mae,
#                 'time_seconds': time.time() - start_time
#             })
#
#             print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, Time: {time.time() - start_time:.2f}s")
#
#         except Exception as e:
#             print(f"  Error with {sim_options}, k={k}: {str(e)}")
#             knn_results.append({
#                 'sim_name': sim_options['name'],
#                 'user_based': sim_options['user_based'],
#                 'k': k,
#                 'rmse': float('inf'),
#                 'mae': float('inf'),
#                 'time_seconds': time.time() - start_time
#             })
#
# Convert results to DataFrame
# knn_results_df = pd.DataFrame(knn_results)
# print("\nInitial configuration results:")
# print(knn_results_df.sort_values(by='rmse'))
#
# Find the best similarity method
# best_initial = knn_results_df.sort_values(by='rmse').iloc[0]
# best_sim_options = {
#     'name': best_initial['sim_name'],
#     'user_based': best_initial['user_based']
# }
#
# Now test more k values with the best similarity method
# print(f"\nTesting more k values with best similarity method: {best_sim_options}...")
# extended_k_values = [20, 30, 40, 50, 60]
# extended_results = []
#
# for k in extended_k_values:
#     print(f"Testing k={k}")
#     start_time = time.time()
#
#     try:
#         knn = KNNWithMeans(k=k, sim_options=best_sim_options, verbose=False)
#         knn.fit(trainset)
#         predictions = knn.test(sampled_testset)
#         rmse = accuracy.rmse(predictions, verbose=False)
#         mae = accuracy.mae(predictions, verbose=False)
#
#         extended_results.append({
#             'sim_name': best_sim_options['name'],
#             'user_based': best_sim_options['user_based'],
#             'k': k,
#             'rmse': rmse,
#             'mae': mae,
#             'time_seconds': time.time() - start_time
#         })
#
#         print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, Time: {time.time() - start_time:.2f}s")
#
#     except Exception as e:
#         print(f"  Error with k={k}: {str(e)}")
#
# Combine all results
# all_results_df = pd.concat([knn_results_df, pd.DataFrame(extended_results)], ignore_index=True)
# print("\nAll configurations with performance metrics:")
# print(all_results_df.sort_values(by='rmse'))
#
# print("\nBest KNN configuration (by RMSE):")
# print(all_results_df.sort_values(by='rmse').head(1))
#
# Train the best KNN model on the full dataset
# best_knn = all_results_df.sort_values(by='rmse').iloc[0]
# best_sim_options = {
#     'name': best_knn['sim_name'],
#     'user_based': best_knn['user_based']
# }
# knn_model = KNNWithMeans(k=int(best_knn['k']), sim_options=best_sim_options)
# print(f"\nTraining final model with {best_sim_options}, k={int(best_knn['k'])}...")
# knn_model.fit(trainset)
#
# Evaluate the best KNN model on the full test set
# print("Evaluating on full test set...")
# predictions_knn = knn_model.test(testset)
# rmse_knn = accuracy.rmse(predictions_knn, verbose=True)
# mae_knn = accuracy.mae(predictions_knn, verbose=True)

# %%
# Train KNN model with reasonable defaults (skip grid search for speed)
print("Training KNN model...")
knn_sim_options = {
    "name": "cosine",
    "user_based": False,  # Item-based CF
}
knn_model = KNNWithMeans(k=40, sim_options=knn_sim_options, verbose=False)
knn_model.fit(trainset)

# Evaluate KNN model
print("Evaluating KNN model...")
predictions_knn = knn_model.test(testset)
rmse_knn = accuracy.rmse(predictions_knn, verbose=True)
mae_knn = accuracy.mae(predictions_knn, verbose=True)

print(f"\nKNN Model Performance: RMSE={rmse_knn:.4f}, MAE={mae_knn:.4f}")

# %% [markdown]
# <a id="topn"></a>
# ### 4.3 Generating Top-N Recommendations
#
# After training, we can request **predicted ratings** for every item for a specific user. The Surprise library has a handy example function:


# %%
def recommend_articles_for_user_knn(predictions, n=5):
    """
    Return the top-N recommendations for each user from a set of predictions.
    """
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sort predictions for each user and retrieve the top n
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# Let's compute top-n for all users in the test set:
top_n_svd = recommend_articles_for_user_knn(predictions_svd, n=20)
top_n_knn = recommend_articles_for_user_knn(predictions_knn, n=20)

# Example: top recommendations for user_id=23
print("Top SVD recommendations for user_id=23:")
print(top_n_svd.get(23))
print("\nTop KNN recommendations for user_id=23:")
print(top_n_knn.get(23))


# %%
def recommend_articles_for_user_surprise(user_id, model, all_items, n=5, filter_items=None):
    """
    Generate top-n recommendations for a single user using a Surprise model.
    1) Predict rating for each item for the user.
    2) Sort by predicted rating, return top n articles.

    Args:
        user_id: the user for whom we want recommendations
        model: a Surprise model (with .predict)
        all_items: all possible item_ids
        n: number of recommendations
        filter_items: items to exclude from recommendations (e.g., already seen items)

    Returns:
        list of top n (article_id, estimated_rating)
    """
    if filter_items is None:
        filter_items = set()

    preds = []
    for item_id in all_items:
        if item_id in filter_items:
            continue
        try:
            pred = model.predict(uid=user_id, iid=item_id)
            preds.append((item_id, pred.est))
        except:
            # Skip items that cause prediction errors
            continue

    # Sort by rating desc
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    return preds_sorted[:n]


# Get user's already seen articles
def get_user_seen_articles(user_id, clicks_df):
    return set(clicks_df[clicks_df["user_id"] == user_id]["click_article_id"])


# Example usage
unique_article_ids = user_article_ratings["click_article_id"].unique()
test_user_id = 23
seen_articles = get_user_seen_articles(test_user_id, df_clicks_full)

print(f"User {test_user_id} has seen {len(seen_articles)} articles")

# Get SVD recommendations
svd_recommendations = recommend_articles_for_user_surprise(user_id=test_user_id, model=svd_model, all_items=unique_article_ids, n=5, filter_items=seen_articles)
print("\nSVD Recommendations for user_id=23:")
print(svd_recommendations)

# Get KNN recommendations
knn_recommendations = recommend_articles_for_user_surprise(user_id=test_user_id, model=knn_model, all_items=unique_article_ids, n=5, filter_items=seen_articles)
print("\nKNN Recommendations for user_id=23:")
print(knn_recommendations)


# %% [markdown]
# <a id="hybrid-approach"></a>
# ## 5. Hybrid Approach
#
# We now have:
# - **Content-based** suggestions: "Given an article, find similar articles."
# - **Collaborative-based** suggestions: "Given a user, find items that similar users liked."
#
# A hybrid approach combines the strengths of both methods.

# %% [markdown]
# <a id="weighted-ensemble"></a>
# ### 5.1 Weighted Ensemble Method
#
# We can combine recommendations from different models using a weighted approach:


# %%
def get_hybrid_recommendations(user_id, clicks_df, embeddings_matrix, cf_model, all_items, n=5, cb_weight=0.5, cf_weight=0.5, cb_method="combined"):
    """
    Generate hybrid recommendations by combining content-based and collaborative filtering.

    Args:
        user_id: User ID
        clicks_df: DataFrame with click data
        embeddings_matrix: Matrix of article embeddings
        cf_model: Trained collaborative filtering model (Surprise)
        all_items: All possible article IDs
        n: Number of recommendations to return
        cb_weight: Weight for content-based recommendations (0-1)
        cf_weight: Weight for collaborative filtering recommendations (0-1)
        cb_method: Content-based method to use ('recent', 'average', 'combined')

    Returns:
        List of recommended article IDs
    """
    # Get user's already seen articles
    seen_articles = get_user_seen_articles(user_id, clicks_df)

    # Get content-based recommendations
    if cb_method == "recent":
        cb_recs = get_content_recommendations_recent(user_id, clicks_df, embeddings_matrix, top_n=n * 2)
    elif cb_method == "average":
        cb_recs = get_content_recommendations_average(user_id, clicks_df, embeddings_matrix, top_n=n * 2)
    else:  # 'combined'
        cb_recs = get_content_recommendations_combined(user_id, clicks_df, embeddings_matrix, top_n=n * 2)

    # Convert to dict for easy scoring
    cb_scores = {art_id: (n * 2 - i) / (n * 2) for i, art_id in enumerate(cb_recs)}

    # Get collaborative filtering recommendations
    cf_recs = recommend_articles_for_user_surprise(user_id, cf_model, all_items, n=n * 2, filter_items=seen_articles)

    # Normalize CF scores to 0-1 range
    if cf_recs:
        max_cf_score = max([score for _, score in cf_recs])
        min_cf_score = min([score for _, score in cf_recs])
        score_range = max_cf_score - min_cf_score

        if score_range > 0:
            cf_scores = {art_id: (score - min_cf_score) / score_range for art_id, score in cf_recs}
        else:
            cf_scores = {art_id: 1.0 for art_id, _ in cf_recs}
    else:
        cf_scores = {}

    # Combine scores
    combined_scores = {}
    all_art_ids = set(cb_scores.keys()) | set(cf_scores.keys())

    for art_id in all_art_ids:
        cb_score = cb_scores.get(art_id, 0)
        cf_score = cf_scores.get(art_id, 0)
        combined_scores[art_id] = cb_weight * cb_score + cf_weight * cf_score

    # Sort by combined score and return top n
    recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n]

    return [art_id for art_id, _ in recommendations]


# Example usage
test_user_id = 23
hybrid_recs = get_hybrid_recommendations(test_user_id, df_clicks_full, articles_embeddings, svd_model, unique_article_ids, n=5, cb_weight=0.6, cf_weight=0.4)

print(f"Hybrid recommendations for user {test_user_id}:")
print(hybrid_recs)

# Compare with pure content-based and collaborative filtering
print("\nContent-based recommendations:")
print(get_content_recommendations_combined(test_user_id, df_clicks_full, articles_embeddings))

print("\nCollaborative filtering recommendations:")
seen_articles = get_user_seen_articles(test_user_id, df_clicks_full)
cf_recs = recommend_articles_for_user_surprise(test_user_id, svd_model, unique_article_ids, n=5, filter_items=seen_articles)
print([art_id for art_id, _ in cf_recs])


# %% [markdown]
# <a id="user-based-selection"></a>
# ### 5.2 User-Based Selection Strategy
#
# For different types of users, we might want to use different recommendation strategies:


# %%
def get_user_activity_level(user_id, clicks_df):
    """
    Determine user activity level based on click history.

    Returns:
        'new': User with very few clicks (1-2)
        'casual': User with moderate activity (3-10 clicks)
        'active': User with significant history (>10 clicks)
    """
    user_clicks = clicks_df[clicks_df["user_id"] == user_id]
    num_clicks = len(user_clicks)

    if num_clicks <= 2:
        return "new"
    elif num_clicks <= 10:
        return "casual"
    else:
        return "active"


def get_adaptive_recommendations(user_id, clicks_df, embeddings_matrix, cf_model, all_items, n=5):
    """
    Adapt recommendation strategy based on user activity level.

    - New users: Mostly content-based (80% CB, 20% CF)
    - Casual users: Balanced approach (50% CB, 50% CF)
    - Active users: More collaborative filtering (30% CB, 70% CF)
    """
    activity_level = get_user_activity_level(user_id, clicks_df)

    if activity_level == "new":
        return get_hybrid_recommendations(user_id, clicks_df, embeddings_matrix, cf_model, all_items, n=n, cb_weight=0.8, cf_weight=0.2, cb_method="recent")
    elif activity_level == "casual":
        return get_hybrid_recommendations(user_id, clicks_df, embeddings_matrix, cf_model, all_items, n=n, cb_weight=0.5, cf_weight=0.5, cb_method="combined")
    else:  # 'active'
        return get_hybrid_recommendations(user_id, clicks_df, embeddings_matrix, cf_model, all_items, n=n, cb_weight=0.3, cf_weight=0.7, cb_method="average")


# Test on users with different activity levels
# Find users with different activity levels
user_activity = df_clicks_full.groupby("user_id").size().reset_index()
user_activity.columns = ["user_id", "num_clicks"]

new_user = user_activity[user_activity["num_clicks"] <= 2].iloc[0]["user_id"]
casual_user = user_activity[(user_activity["num_clicks"] > 2) & (user_activity["num_clicks"] <= 10)].iloc[0]["user_id"]
active_user = user_activity[user_activity["num_clicks"] > 10].iloc[0]["user_id"]

print("Testing adaptive recommendations on different user types:")
print(f"New user ({new_user}) recommendations:")
print(get_adaptive_recommendations(new_user, df_clicks_full, articles_embeddings, svd_model, unique_article_ids))

print(f"\nCasual user ({casual_user}) recommendations:")
print(get_adaptive_recommendations(casual_user, df_clicks_full, articles_embeddings, svd_model, unique_article_ids))

print(f"\nActive user ({active_user}) recommendations:")
print(get_adaptive_recommendations(active_user, df_clicks_full, articles_embeddings, svd_model, unique_article_ids))


# %% [markdown]
# <a id="hybrid-evaluation"></a>
# ### 5.3 Evaluation of Hybrid Approach
#
# To evaluate our hybrid recommendation approach, we can use a leave-one-out strategy:
# 1. For users with sufficient history, hide their most recent click
# 2. Generate recommendations
# 3. Check if the hidden article appears in the recommendations


# %%
def evaluate_recommendations(clicks_df, embeddings_matrix, cf_model, all_items, num_users=100, n=5):
    """
    Evaluate recommendation quality using a leave-one-out approach.

    Args:
        clicks_df: DataFrame with click data
        embeddings_matrix: Matrix of article embeddings
        cf_model: Collaborative filtering model
        all_items: All possible article IDs
        num_users: Number of users to evaluate on
        n: Number of recommendations to generate

    Returns:
        Dictionary with evaluation metrics
    """
    # Find users with at least 2 clicks (so we can hide one)
    user_counts = clicks_df.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= 2].index.tolist()

    # Sample users for evaluation
    if len(eligible_users) > num_users:
        test_users = random.sample(eligible_users, num_users)
    else:
        test_users = eligible_users

    results = {"cb_recent_hits": 0, "cb_combined_hits": 0, "cf_hits": 0, "hybrid_hits": 0, "adaptive_hits": 0, "total_users": len(test_users)}

    for user_id in test_users:
        # Get user's clicks, sorted by timestamp
        user_clicks = clicks_df[clicks_df["user_id"] == user_id].sort_values(by="click_timestamp", ascending=False)

        # Hold out the most recent click
        most_recent_click = user_clicks.iloc[0]["click_article_id"]
        training_clicks = user_clicks.iloc[1:]

        # Create a modified clicks DataFrame without the most recent click
        modified_clicks = clicks_df[~((clicks_df["user_id"] == user_id) & (clicks_df["click_article_id"] == most_recent_click))]

        # Generate recommendations using different methods
        cb_recent_recs = get_content_recommendations_recent(user_id, modified_clicks, embeddings_matrix, top_n=n)

        cb_combined_recs = get_content_recommendations_combined(user_id, modified_clicks, embeddings_matrix, top_n=n)

        seen_articles = set(training_clicks["click_article_id"])
        cf_recs = recommend_articles_for_user_surprise(user_id, cf_model, all_items, n=n, filter_items=seen_articles)
        cf_rec_ids = [art_id for art_id, _ in cf_recs]

        hybrid_recs = get_hybrid_recommendations(user_id, modified_clicks, embeddings_matrix, cf_model, all_items, n=n)

        adaptive_recs = get_adaptive_recommendations(user_id, modified_clicks, embeddings_matrix, cf_model, all_items, n=n)

        # Check if the held-out article is in the recommendations
        if most_recent_click in cb_recent_recs:
            results["cb_recent_hits"] += 1

        if most_recent_click in cb_combined_recs:
            results["cb_combined_hits"] += 1

        if most_recent_click in cf_rec_ids:
            results["cf_hits"] += 1

        if most_recent_click in hybrid_recs:
            results["hybrid_hits"] += 1

        if most_recent_click in adaptive_recs:
            results["adaptive_hits"] += 1

    # Calculate hit rates
    for key in ["cb_recent_hits", "cb_combined_hits", "cf_hits", "hybrid_hits", "adaptive_hits"]:
        results[key + "_rate"] = results[key] / results["total_users"]

    return results


# Debug: Check one user to verify recommendation functions work
debug_user = df_clicks_full.groupby("user_id").size().sort_values(ascending=False).index[0]
debug_clicks = df_clicks_full[df_clicks_full["user_id"] == debug_user].sort_values("click_timestamp", ascending=False)
print(f"Debug user {debug_user} has {len(debug_clicks)} clicks")
most_recent = debug_clicks.iloc[0]["click_article_id"]
print(f"Most recent article: {most_recent} (type: {type(most_recent)})")

debug_recs = get_content_recommendations_recent(debug_user, df_clicks_full, articles_embeddings, top_n=10)
print(f"CB Recent recommendations (type={type(debug_recs)}): {debug_recs[:5]}")
print(f"CB rec element type: {type(debug_recs[0]) if debug_recs else 'empty'}")

debug_cf_recs = recommend_articles_for_user_surprise(debug_user, svd_model, unique_article_ids, n=10)
cf_ids = [art_id for art_id, _ in debug_cf_recs]
print(f"CF recommendations: {cf_ids[:5]}")
print(f"CF rec element type: {type(cf_ids[0]) if cf_ids else 'empty'}")

# Check if user is in SVD training data
try:
    test_pred = svd_model.predict(debug_user, most_recent)
    print(f"SVD can predict for this user: rating={test_pred.est:.2f}")
except:
    print("WARNING: User not in SVD training data!")

# Evaluate our recommendation methods
# Note: With 364K articles, we need large n to get meaningful hit rates
# n=100 gives ~0.03% coverage per user
evaluation_results = evaluate_recommendations(df_clicks_full, articles_embeddings, svd_model, unique_article_ids, num_users=200, n=100)

# Interpretation:
# - Hit Rate is strict with large catalogs (364K articles)
# - n=100 recommendations = 0.03% of catalog
# - Any hit rate > 0.03% indicates model is better than random

# Display results
print("Recommendation Evaluation Results (Hit Rate):")
print(f"Content-Based (Recent): {evaluation_results['cb_recent_hits_rate']:.4f}")
print(f"Content-Based (Combined): {evaluation_results['cb_combined_hits_rate']:.4f}")
print(f"Collaborative Filtering: {evaluation_results['cf_hits_rate']:.4f}")
print(f"Hybrid Approach: {evaluation_results['hybrid_hits_rate']:.4f}")
print(f"Adaptive Approach: {evaluation_results['adaptive_hits_rate']:.4f}")

# Save evaluation report
eval_report = f"""Recommendation Evaluation Results
==================================
Evaluation Method: Leave-One-Out
Number of Users Tested: {evaluation_results["total_users"]}

Hit Rates (higher is better):
-----------------------------
Content-Based (Recent):   {evaluation_results["cb_recent_hits_rate"]:.4f} ({evaluation_results["cb_recent_hits"]}/{evaluation_results["total_users"]} hits)
Content-Based (Combined): {evaluation_results["cb_combined_hits_rate"]:.4f} ({evaluation_results["cb_combined_hits"]}/{evaluation_results["total_users"]} hits)
Collaborative Filtering:  {evaluation_results["cf_hits_rate"]:.4f} ({evaluation_results["cf_hits"]}/{evaluation_results["total_users"]} hits)
Hybrid Approach:          {evaluation_results["hybrid_hits_rate"]:.4f} ({evaluation_results["hybrid_hits"]}/{evaluation_results["total_users"]} hits)
Adaptive Approach:        {evaluation_results["adaptive_hits_rate"]:.4f} ({evaluation_results["adaptive_hits"]}/{evaluation_results["total_users"]} hits)

Best Performing Method: {"Adaptive" if evaluation_results["adaptive_hits_rate"] >= max(evaluation_results["hybrid_hits_rate"], evaluation_results["cf_hits_rate"]) else "Hybrid" if evaluation_results["hybrid_hits_rate"] >= evaluation_results["cf_hits_rate"] else "CF"}
"""
save_report(eval_report, "recommendation_evaluation.txt")

# Plot comparison
methods = ["Content-Based\n(Recent)", "Content-Based\n(Combined)", "Collaborative\nFiltering", "Hybrid\nApproach", "Adaptive\nApproach"]
hit_rates = [
    evaluation_results["cb_recent_hits_rate"],
    evaluation_results["cb_combined_hits_rate"],
    evaluation_results["cf_hits_rate"],
    evaluation_results["hybrid_hits_rate"],
    evaluation_results["adaptive_hits_rate"],
]

plt.figure(figsize=(12, 6))
bars = plt.bar(methods, hit_rates, color=["lightblue", "skyblue", "lightgreen", "orange", "salmon"])

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.4f}", ha="center", va="bottom")

plt.title("Recommendation Method Comparison (Hit Rate)")
plt.ylabel("Hit Rate")
plt.ylim(0, max(hit_rates) * 1.2)  # Add some space above bars
plt.tight_layout()
save_plot("08_recommendation_comparison.png")
plt.show()


# %%
# Final recommendations function for deployment
def get_recommendations_for_user(user_id, clicks_df, embeddings_matrix, cf_model, all_items, n=5):
    """
    Get recommendations for a user using the adaptive hybrid approach.
    This is the main function that would be deployed in production.

    Args:
        user_id: User ID
        clicks_df: DataFrame with click data
        embeddings_matrix: Matrix of article embeddings
        cf_model: Collaborative filtering model
        all_items: All possible article IDs
        n: Number of recommendations to return

    Returns:
        List of recommended article IDs
    """
    # Check if user exists in the dataset
    if user_id not in clicks_df["user_id"].values:
        # For new users, recommend popular articles
        article_popularity = clicks_df["click_article_id"].value_counts()
        return article_popularity.head(n).index.tolist()

    # For existing users, use the adaptive approach
    return get_adaptive_recommendations(user_id, clicks_df, embeddings_matrix, cf_model, all_items, n=n)


# Example usage for production
user_id = 42  # Example user
recommendations = get_recommendations_for_user(user_id, df_clicks_full, articles_embeddings, svd_model, unique_article_ids)

print(f"Final recommendations for user {user_id}:")
print(recommendations)

# %% [markdown]
# ## 6. Saving Models for Deployment
#
# The embeddings have already been reduced via PCA in section 3.1.1 (250 → 50 dimensions).
# This optimization is crucial for Azure Functions deployment (1.5 GB memory limit).

# %%
from surprise import dump


def save_models(svd_model, articles_df, article_embeddings, user_article_ratings, output_folder="models", knn_model=None):
    """
    Save all models and required data to files for later loading.

    Args:
        svd_model: Trained SVD model from Surprise
        articles_df: DataFrame with article metadata
        article_embeddings: NumPy array of article embeddings
        user_article_ratings: DataFrame with user-article ratings
        output_folder: Folder to save models to
        knn_model: Optional trained KNN model from Surprise
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # 1. Save SVD model using Surprise's built-in dump function
    dump.dump(os.path.join(output_folder, "svd_model.pkl"), algo=svd_model)
    print(f"SVD model saved to {output_folder}/svd_model.pkl")

    # 2. Save KNN model (optional)
    if knn_model is not None:
        dump.dump(os.path.join(output_folder, "knn_model.pkl"), algo=knn_model)
        print(f"KNN model saved to {output_folder}/knn_model.pkl")

    # 3. Save article embeddings
    with open(os.path.join(output_folder, "article_embeddings.pkl"), "wb") as f:
        pickle.dump(article_embeddings, f)
    print(f"Article embeddings saved to {output_folder}/article_embeddings.pkl")

    # 4. Save article metadata (only essential columns)
    essential_article_cols = ["article_id", "category_id", "publisher_id"]
    articles_df[essential_article_cols].to_csv(os.path.join(output_folder, "articles_metadata.csv"), index=False)
    print(f"Article metadata saved to {output_folder}/articles_metadata.csv")

    # 5. Save a mapping of article IDs to indices for embeddings access
    article_ids = articles_df["article_id"].values
    article_id_to_idx = {art_id: idx for idx, art_id in enumerate(article_ids)}
    with open(os.path.join(output_folder, "article_id_mapping.pkl"), "wb") as f:
        pickle.dump(article_id_to_idx, f)
    print(f"Article ID mapping saved to {output_folder}/article_id_mapping.pkl")

    # 6. Save user-article ratings for cold-start recommendations
    user_article_ratings.to_csv(os.path.join(output_folder, "user_article_ratings.csv"), index=False)
    print(f"User-article ratings saved to {output_folder}/user_article_ratings.csv")

    # 7. Save popularity rankings for cold-start recommendations
    article_popularity = user_article_ratings.groupby("click_article_id")["rating"].sum().reset_index()
    article_popularity = article_popularity.sort_values("rating", ascending=False)
    article_popularity.to_csv(os.path.join(output_folder, "article_popularity.csv"), index=False)
    print(f"Article popularity rankings saved to {output_folder}/article_popularity.csv")


# %%
# Save models to files for deployment
# Note: articles_embeddings is already PCA-reduced (50 dims) from section 3.1.1
save_models(
    svd_model=svd_model,
    articles_df=articles_df,
    article_embeddings=articles_embeddings,  # Already PCA-reduced in section 3.1.1
    user_article_ratings=user_article_ratings,
    output_folder="models",
    knn_model=knn_model,
)

# %% [markdown]
# ### Deployment Summary
#
# | Optimization | Before | After | Savings |
# |--------------|--------|-------|---------|
# | Embeddings (PCA 250→50) | 347 MB | 69 MB | **80%** |
# | Total model size | 546 MB | 268 MB | **51%** |
# | Variance retained | 100% | 95% | Minimal loss |
#
# The PCA-reduced embeddings (applied in section 3.1.1) enable deployment on
# Azure Functions Consumption Plan (~$1-5/month for demo usage).
