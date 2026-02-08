"""
Model storage container for recommendation models.
"""

from typing import Any

import numpy as np
import pandas as pd


class ModelStore:
    """Container for recommendation models"""

    def __init__(self):
        # Collaborative filtering model
        self.svd_model: Any | None = None

        # Content-based recommendation data
        self.article_embeddings: np.ndarray | None = None
        self.article_id_to_idx: dict[int, int] | None = None
        self.idx_to_article_id: dict[int, int] | None = None

        # Popularity and user data
        self.article_popularity: pd.DataFrame | None = None
        self.user_article_ratings: pd.DataFrame | None = None

        # Metadata
        self.articles_metadata: pd.DataFrame | None = None

        # Loading status
        self.models_loaded: bool = False
        self.loaded_models: dict[str, bool] = {
            "svd_model": False,
            "article_embeddings": False,
            "article_id_mapping": False,
            "article_popularity": False,
            "user_article_ratings": False,
            "articles_metadata": False,
        }

    def is_ready(self) -> bool:
        """Check if required models are loaded for basic functionality"""
        return self.svd_model is not None

    def is_content_ready(self) -> bool:
        """Check if content-based recommendation models are loaded"""
        return self.article_embeddings is not None and self.article_id_to_idx is not None

    def is_hybrid_ready(self) -> bool:
        """Check if models for hybrid recommendations are loaded"""
        return self.is_ready() and self.is_content_ready()

    def get_status(self) -> dict[str, bool]:
        """Get loading status of all models"""
        self.loaded_models = {
            "svd_model": self.svd_model is not None,
            "article_embeddings": self.article_embeddings is not None,
            "article_id_mapping": self.article_id_to_idx is not None,
            "article_popularity": self.article_popularity is not None,
            "user_article_ratings": self.user_article_ratings is not None,
            "articles_metadata": self.articles_metadata is not None,
        }
        return self.loaded_models


# Create a singleton instance for import
model_store = ModelStore()
